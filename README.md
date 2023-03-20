# AST
This is unofficial implementation of "Asymmetric student-teacher networks for industrial anomaly detection"

1. Write your data directory in config.py 
2. set config.py 
3. train_teacher.py 
4. train_student.py
5. eval.py 

# Result 
## MVtecAD Image-level AUROC 
|  | mean | max | Paper |
| --- | --- | --- | --- |
| leather | 1 | 1 | 1 |
| zipper | 0.991 | 0.977 | 0.991 |
| metal_nut | 0.989 | 0.996 | 0.985 |
| wood | 0.988 | 0.992 | 1 |
| pill | 0.992 | 0.964 | 0.991 |
| transistor | 0.990 | 0.987 | 0.993 |
| grid | 0.990 | 0.999 | 0.991 |
| tile | 0.999 | 0.996 | 1 |
| capsule | 0.992 | 0.971 | 0.997 |
| hazelnut | 0.998 | 0.997 | 1 |
| toothbrush | 0.961 | 0.864 | 0.966 |
| screw | 0.993 | 0.944 | 0.997 |
| carpet | 0.972 | 0.972 | 0.975 |
| bottle | 0.998 | 0.994 | 1 |
| cable | 0.992 | 0.939 | 0.985 |