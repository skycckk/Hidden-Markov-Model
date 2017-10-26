from sklearn import svm
from sklearn.feature_selection import RFE

with open('malwareBenignScores.txt') as f:
    read_data = f.readlines()

content = [x.strip().split() for x in read_data]
sample_start_row = 2
malware_start_col = 1
benign_start_col = 4

malware_samples = list()
for i in range(sample_start_row, sample_start_row + 40):
    feature = list()
    [feature.append(float(content[i][malware_start_col + j])) for j in range(3)]
    malware_samples.append(feature)
print(len(malware_samples))

benign_samples = list()
for i in range(sample_start_row, sample_start_row + 40):
    feature = list()
    [feature.append(float(content[i][benign_start_col + j])) for j in range(3)]
    benign_samples.append(feature)
print(len(benign_samples))

ground_truth = ['malware', 'benign']
training_set = malware_samples[:20] + benign_samples[:20]
testing_set = malware_samples[20:] + benign_samples[20:]
training_labels = ['m'] * 20 + ['b'] * 20
testing_labels = ['m'] * 20 + ['b'] * 20

clf = svm.SVC(C=3, kernel='poly', degree=4)
# clf = svm.SVC(kernel='linear')

clf.fit(training_set, training_labels)
# print("Feature weights: ", clf.coef_)
p_res = clf.predict(testing_set)

accuracy = 0
for i in range(len(p_res)):
    if p_res[i] == testing_labels[i]:
        accuracy += 1

accuracy = (accuracy / len(p_res)) * 100
print("Accuracy:", accuracy)
print(p_res)

# Linear SVM-RFE
# estimator = svm.SVC(kernel='linear')
# clf_reduced = RFE(estimator, 1, step=1, verbose=1)
# clf_reduced = clf_reduced.fit(training_set, training_labels)