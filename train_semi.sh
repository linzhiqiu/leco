python train_semi.py --train_mode none_0_scratch_linear_1_scratch_linear --hparam_strs cifar_1_batch_128
python train_semi.py --train_mode none_0_scratch_linear_1_scratch_linear --hparam_strs cifar_1_batch_128 --partial_feedback_mode single_head 
python train_semi.py --train_mode none_0_scratch_linear_1_scratch_linear --hparam_strs cifar_1_batch_128 --partial_feedback_mode two_head

python train_semi.py --train_mode none_0_scratch_linear_1_scratch_linear --hparam_strs cifar_1_batch_128 --semi_supervised_alg PL --pl_threshold 0.95
python train_semi.py --train_mode none_0_scratch_linear_1_scratch_linear --hparam_strs cifar_1_batch_128 --semi_supervised_alg PL --pl_threshold 0.95 --partial_feedback_mode single_head
python train_semi.py --train_mode none_0_scratch_linear_1_scratch_linear --hparam_strs cifar_1_batch_128 --semi_supervised_alg PL --pl_threshold 0.95 --partial_feedback_mode two_head
python train_semi.py --train_mode none_0_scratch_linear_1_scratch_linear --hparam_strs cifar_1_batch_128 --semi_supervised_alg PL --pl_threshold 0.95 --hierarchical_ssl filtering
python train_semi.py --train_mode none_0_scratch_linear_1_scratch_linear --hparam_strs cifar_1_batch_128 --semi_supervised_alg PL --pl_threshold 0.95 --hierarchical_ssl filtering_conditioning
python train_semi.py --train_mode none_0_scratch_linear_1_scratch_linear --hparam_strs cifar_1_batch_128 --semi_supervised_alg PL --pl_threshold 0.95 --hierarchical_ssl conditioning
python train_semi.py --train_mode none_0_scratch_linear_1_scratch_linear --hparam_strs cifar_1_batch_128 --semi_supervised_alg PL --pl_threshold 0.95 --partial_feedback_mode single_head --hierarchical_ssl filtering
python train_semi.py --train_mode none_0_scratch_linear_1_scratch_linear --hparam_strs cifar_1_batch_128 --semi_supervised_alg PL --pl_threshold 0.95 --partial_feedback_mode single_head --hierarchical_ssl filtering_conditioning
python train_semi.py --train_mode none_0_scratch_linear_1_scratch_linear --hparam_strs cifar_1_batch_128 --semi_supervised_alg PL --pl_threshold 0.95 --partial_feedback_mode single_head --hierarchical_ssl conditioning
python train_semi.py --train_mode none_0_scratch_linear_1_scratch_linear --hparam_strs cifar_1_batch_128 --semi_supervised_alg PL --pl_threshold 0.95 --partial_feedback_mode two_head --hierarchical_ssl filtering
python train_semi.py --train_mode none_0_scratch_linear_1_scratch_linear --hparam_strs cifar_1_batch_128 --semi_supervised_alg PL --pl_threshold 0.95 --partial_feedback_mode two_head --hierarchical_ssl filtering_conditioning
python train_semi.py --train_mode none_0_scratch_linear_1_scratch_linear --hparam_strs cifar_1_batch_128 --semi_supervised_alg PL --pl_threshold 0.95 --partial_feedback_mode two_head --hierarchical_ssl conditioning

python train_semi.py --train_mode none_0_scratch_linear_1_scratch_linear --hparam_strs cifar_1_batch_128 --semi_supervised_alg DistillHard
python train_semi.py --train_mode none_0_scratch_linear_1_scratch_linear --hparam_strs cifar_1_batch_128 --semi_supervised_alg DistillHard --hierarchical_ssl filtering
python train_semi.py --train_mode none_0_scratch_linear_1_scratch_linear --hparam_strs cifar_1_batch_128 --semi_supervised_alg DistillHard --hierarchical_ssl filtering_conditioning
python train_semi.py --train_mode none_0_scratch_linear_1_scratch_linear --hparam_strs cifar_1_batch_128 --semi_supervised_alg DistillHard --hierarchical_ssl conditioning
python train_semi.py --train_mode none_0_scratch_linear_1_scratch_linear --hparam_strs cifar_1_batch_128 --semi_supervised_alg DistillHard --partial_feedback_mode single_head
python train_semi.py --train_mode none_0_scratch_linear_1_scratch_linear --hparam_strs cifar_1_batch_128 --semi_supervised_alg DistillHard --partial_feedback_mode two_head
python train_semi.py --train_mode none_0_scratch_linear_1_scratch_linear --hparam_strs cifar_1_batch_128 --semi_supervised_alg DistillHard --partial_feedback_mode single_head --hierarchical_ssl filtering
python train_semi.py --train_mode none_0_scratch_linear_1_scratch_linear --hparam_strs cifar_1_batch_128 --semi_supervised_alg DistillHard --partial_feedback_mode single_head --hierarchical_ssl filtering_conditioning
python train_semi.py --train_mode none_0_scratch_linear_1_scratch_linear --hparam_strs cifar_1_batch_128 --semi_supervised_alg DistillHard --partial_feedback_mode single_head --hierarchical_ssl conditioning
python train_semi.py --train_mode none_0_scratch_linear_1_scratch_linear --hparam_strs cifar_1_batch_128 --semi_supervised_alg DistillHard --partial_feedback_mode two_head --hierarchical_ssl filtering
python train_semi.py --train_mode none_0_scratch_linear_1_scratch_linear --hparam_strs cifar_1_batch_128 --semi_supervised_alg DistillHard --partial_feedback_mode two_head --hierarchical_ssl filtering_conditioning
python train_semi.py --train_mode none_0_scratch_linear_1_scratch_linear --hparam_strs cifar_1_batch_128 --semi_supervised_alg DistillHard --partial_feedback_mode two_head --hierarchical_ssl conditioning

python train_semi.py --train_mode none_0_scratch_linear_1_scratch_linear --hparam_strs cifar_1_batch_128 --semi_supervised_alg DistillSoft
python train_semi.py --train_mode none_0_scratch_linear_1_scratch_linear --hparam_strs cifar_1_batch_128 --semi_supervised_alg DistillSoft --hierarchical_ssl filtering
python train_semi.py --train_mode none_0_scratch_linear_1_scratch_linear --hparam_strs cifar_1_batch_128 --semi_supervised_alg DistillSoft --hierarchical_ssl filtering_conditioning
python train_semi.py --train_mode none_0_scratch_linear_1_scratch_linear --hparam_strs cifar_1_batch_128 --semi_supervised_alg DistillSoft --hierarchical_ssl conditioning
python train_semi.py --train_mode none_0_scratch_linear_1_scratch_linear --hparam_strs cifar_1_batch_128 --semi_supervised_alg DistillSoft --partial_feedback_mode single_head
python train_semi.py --train_mode none_0_scratch_linear_1_scratch_linear --hparam_strs cifar_1_batch_128 --semi_supervised_alg DistillSoft --partial_feedback_mode two_head
python train_semi.py --train_mode none_0_scratch_linear_1_scratch_linear --hparam_strs cifar_1_batch_128 --semi_supervised_alg DistillSoft --partial_feedback_mode single_head --hierarchical_ssl filtering
python train_semi.py --train_mode none_0_scratch_linear_1_scratch_linear --hparam_strs cifar_1_batch_128 --semi_supervised_alg DistillSoft --partial_feedback_mode single_head --hierarchical_ssl filtering_conditioning
python train_semi.py --train_mode none_0_scratch_linear_1_scratch_linear --hparam_strs cifar_1_batch_128 --semi_supervised_alg DistillSoft --partial_feedback_mode single_head --hierarchical_ssl conditioning
python train_semi.py --train_mode none_0_scratch_linear_1_scratch_linear --hparam_strs cifar_1_batch_128 --semi_supervised_alg DistillSoft --partial_feedback_mode two_head --hierarchical_ssl filtering
python train_semi.py --train_mode none_0_scratch_linear_1_scratch_linear --hparam_strs cifar_1_batch_128 --semi_supervised_alg DistillSoft --partial_feedback_mode two_head --hierarchical_ssl filtering_conditioning
python train_semi.py --train_mode none_0_scratch_linear_1_scratch_linear --hparam_strs cifar_1_batch_128 --semi_supervised_alg DistillSoft --partial_feedback_mode two_head --hierarchical_ssl conditioning
