python train_semi.py
python train_semi.py --partial_feedback_mode single_head
python train_semi.py --partial_feedback_mode two_head

python train_semi.py --semi_supervised_alg PL --pl_threshold 0.25
python train_semi.py --semi_supervised_alg PL --pl_threshold 0.25 --partial_feedback_mode single_head
python train_semi.py --semi_supervised_alg PL --pl_threshold 0.25 --partial_feedback_mode two_head
python train_semi.py --semi_supervised_alg PL --pl_threshold 0.25 --hierarchical_supervision_mode filtering
python train_semi.py --semi_supervised_alg PL --pl_threshold 0.25 --hierarchical_supervision_mode filtering_conditioning
python train_semi.py --semi_supervised_alg PL --pl_threshold 0.25 --hierarchical_supervision_mode conditioning
python train_semi.py --semi_supervised_alg PL --pl_threshold 0.25 --partial_feedback_mode single_head
python train_semi.py --semi_supervised_alg PL --pl_threshold 0.25 --partial_feedback_mode two_head

python train_semi.py --semi_supervised_alg DistillHard
python train_semi.py --semi_supervised_alg DistillSoft 
python train_semi.py --semi_supervised_alg DistillHard --hierarchical_supervision_mode filtering
python train_semi.py --semi_supervised_alg DistillSoft --hierarchical_supervision_mode filtering 
python train_semi.py --semi_supervised_alg DistillHard --hierarchical_supervision_mode filtering_conditioning
python train_semi.py --semi_supervised_alg DistillSoft --hierarchical_supervision_mode filtering_conditioning
python train_semi.py --semi_supervised_alg DistillHard --hierarchical_supervision_mode conditioning
python train_semi.py --semi_supervised_alg DistillSoft --hierarchical_supervision_mode conditioning
python train_semi.py --semi_supervised_alg DistillHard --partial_feedback_mode single_head
python train_semi.py --semi_supervised_alg DistillSoft --partial_feedback_mode two_head