import AI_hw3 as aihw
import numpy as np

aihw.train()
pred, gt = aihw.test()
assert len(pred) == len(gt), "The length of prediction and ground truth should be the same"
correct = np.count_nonzero(np.array(pred) == np.array(gt))
print("Accuracy: {}%".format(float(correct) / len(pred) * 100))
