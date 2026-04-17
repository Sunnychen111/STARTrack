from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/cps/czl/PDTrack_v3/data/got10k_lmdb'
    settings.got10k_path = '/home/cps/SOT_dataset/got10k/test_data'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_extension_subset_path = '/home/cps/czl/PDTrack_v3/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/cps/czl/PDTrack_v3/data/lasot_lmdb'
    settings.lasot_path = '/home/cps/SOT_dataset/LaSOTBenchmark'
    settings.lasotlang_path = '/home/cps/czl/PDTrack_v3/data/lasot'
    settings.network_path = '/home/cps/czl/PDTrack_v3/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/cps/czl/PDTrack_v3/data/nfs'
    settings.otb_path = '/home/cps/czl/PDTrack_v3/data/OTB2015'
    settings.otblang_path = '/home/cps/czl/PDTrack_v3/data/otb_lang'
    settings.prj_dir = '/home/cps/czl/PDTrack_v3'
    settings.result_plot_path = '/home/cps/czl/PDTrack_v3/test/result_plots'
    settings.results_path = '/home/cps/czl/PDTrack_v3/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/cps/czl/PDTrack_v3'
    settings.segmentation_path = '/home/cps/czl/PDTrack_v3/test/segmentation_results'
    settings.tc128_path = '/home/cps/czl/PDTrack_v3/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/cps/czl/PDTrack_v3/data/tnl2k/test'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/cps/czl/PDTrack_v3/data/trackingnet'
    settings.uav_path = '/home/cps/czl/PDTrack_v3/data/UAV123'
    settings.vot_path = '/home/cps/czl/PDTrack_v3/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

