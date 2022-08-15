CL_MODES = [
    'use_new', # Use the new sampled images
    # 'use_new_fine_for_coarse', # Use the new sampled images for labeled set, but use both new and old images for coarse-supervision (SSL + partial feedback)
    # 'use_new_fine_for_partial_feedback_only', # Use the new sampled images for labeled set, but use both new and old for partial feedback (not ssl)
    'use_t_1_for_multi_task', # (Updated version) Use the new sampled images for labeled set, but use both new and old for partial feedback (not ssl)
    'use_old', # Use the images from previous time
    'use_both', # Use both the new and old images (no semi supervised learning)
    'use_both_for_multi_task', # (Updated version) Use T0+T1 sampled images for labeled set + partial feedback (no ssl)
]