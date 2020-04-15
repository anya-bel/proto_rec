from utils import minimumEditDistance
import numpy as np

distances = []
i = 0
eq = 0
eq_w = 0
total = 0
for word, root, lang in model.test_pairs:
  try:
      pred = model.getRoot(word, lang, model.hidden_size, model.encoder, model.decoder)
  except:
      print('unknown letter: ', word, root)
      i += 1
      continue
  distances.append(minimumEditDistance(root, pred))
  if root == pred:
    eq += 1
  if root == word:
    eq_w += 1
  total += 1
print(total)
print(i)
print("edit", np.mean(distances))
print("acc model", eq/total)
print("baseline", eq_w/total)