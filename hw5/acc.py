from utils.reader import getLabelList
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
predict = getLabelList('saved/seq2seq')
ground = getLabelList('HW5_data/FullLengthVideos/labels/valid')


correct = 0
total = 0
for i in range(len(predict)):
    print(len(predict[i]))
    correct += np.sum(np.array(predict[i])==np.array(ground[i]))
    total += len(predict[i])

print(correct)
print(total)
print(correct/total)


colors = ['w', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'C0', 'C1', 'C2']

ax = plt.subplot(211)
cmap = mpl.colors.ListedColormap(colors)
bounds = [i for i in range(len(colors)+1)]
cb2 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                boundaries=bounds,
                                spacing='proportional',
                                orientation='horizontal')
ax.set_xlabel('labels')
plt.savefig('saved/seq2seq/labels.jpg')
