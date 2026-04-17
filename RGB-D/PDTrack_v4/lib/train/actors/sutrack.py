from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xyxy_to_cxcywh, box_cxcywh_to_xyxy, box_iou
import torch
from lib.train.admin import multigpu
from lib.utils.heapmap_utils import generate_heatmap

from lib.train.loss_dgcl import DepthGuidedContrastiveLossV2

class SUTrack_Actor(BaseActor):
    """ Actor for training the sutrack"""
    def __init__(self, net, objective, loss_weight, settings, cfg):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        self.multi_modal_language = cfg.DATA.MULTI_MODAL_LANGUAGE
        # DGCL loss
        self.dgcl_loss = DepthGuidedContrastiveLossV2(top_k=10, tau_d=0.15, temperature=0.07)
        self.lambda_dgcl = 0.2  # 控制 DGCL loss 占总 loss 的权重

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'search_anno'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        b = data['search_images'].shape[1]   # n,b,c,h,w
        search_list = data['search_images'].view(-1, *data['search_images'].shape[2:]).split(b,dim=0)  # (n*b, c, h, w)
        template_list = data['template_images'].view(-1, *data['template_images'].shape[2:]).split(b,dim=0)
        template_anno_list = data['template_anno'].view(-1, *data['template_anno'].shape[2:]).split(b,dim=0)

        if self.multi_modal_language:
            text = data['nlp_ids'].permute (1,0)
            text_src = self.net(text_data=text, mode='text')
        else:
            text_src = None

        # task_class
        task_index_batch = [self.cfg.MODEL.TASK_INDEX[key.upper()] for key in data['dataset']]
        task_index_batch = torch.tensor(task_index_batch).cuda() #torch.Size([bs])

        # enc_opt = self.net(template_list=template_list,
        #                    search_list=search_list,
        #                    template_anno_list=template_anno_list,
        #                    text_src=text_src,
        #                    task_index=task_index_batch,
        #                    mode='encoder') # forward the encoder
        # outputs, task_class_output = self.net(feature=enc_opt, mode="decoder")
        # # outputs = self.net(feature=enc_opt, mode="decoder")
        # # task_class_output = self.net(feature=enc_opt, mode="task_decoder")
        # task_class_output = task_class_output.view(-1, task_class_output.size(-1))
        # outputs['task_class'] = task_class_output
        # outputs['task_class_label'] = task_index_batch

        # return outputs

        enc_opt = self.net(template_list=template_list,
                           search_list=search_list,
                           template_anno_list=template_anno_list,
                           text_src=text_src,
                           task_index=task_index_batch,
                           mode='encoder') # forward the encoder
                           
        if isinstance(enc_opt, tuple) and len(enc_opt) == 2:
            enc_feat, aux_info = enc_opt
        else:
            enc_feat, aux_info = enc_opt, None

        # [修改] 只把 enc_feat 传给 decoder
        outputs, task_class_output = self.net(feature=enc_feat, mode="decoder")
        
        task_class_output = task_class_output.view(-1, task_class_output.size(-1))
        outputs['task_class'] = task_class_output
        outputs['task_class_label'] = task_index_batch

        # [新增] 将 aux_info 存入 outputs 字典，传给 compute_losses
        if aux_info is not None:
            outputs['aux_info'] = aux_info

        return outputs

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # task classification loss
        task_cls_loss = self.objective['task_cls'](pred_dict['task_class'], pred_dict['task_class_label'])

        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.ENCODER.STRIDE) # list of torch.Size([b, H, W])
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1) # torch.Size([b, 1, H, W])

        # Get boxes
        pred_boxes = pred_dict['pred_boxes'] # torch.Size([b, 1, 4])
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        # weighted sum
        loss = (self.loss_weight['giou'] * giou_loss +
                self.loss_weight['l1'] * l1_loss +
                self.loss_weight['focal'] * location_loss +
                self.loss_weight['task_cls'] * task_cls_loss)

        # if return_status:
        #     # status for log
        #     mean_iou = iou.detach().mean()
        #     status = {"Loss/total": loss.item(),
        #               "Loss/giou": giou_loss.item(),
        #               "Loss/l1": l1_loss.item(),
        #               "Loss/location": location_loss.item(),
        #               "Loss/task_class": task_cls_loss.item(),
        #               "IoU": mean_iou.item()}
        #     return loss, status
        # else:
        #     return loss
        if 'aux_info' in pred_dict:
            # gt_bbox 已经是 (batch, 4) 的归一化坐标 [cx, cy, w, h]，直接封装给 Loss
            targets = {'boxes': gt_bbox}
            loss_dgcl = self.dgcl_loss(pred_dict['aux_info'], targets)
            loss = loss + self.lambda_dgcl * loss_dgcl
        else:
            loss_dgcl = torch.tensor(0.0).to(loss.device)

        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "Loss/task_class": task_cls_loss.item(),
                      "Loss/dgcl": loss_dgcl.item(), # [新增] 记录到 TensorBoard/Log
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss