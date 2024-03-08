import torch

# Make sure your CUDA is available.
assert torch.cuda.is_available()

import numpy as np
import pickle as pkl
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc, roc_curve

import pickle as pkl
f = open('feature_dict_ntu60_cv_r5.pkl', 'rb')
dicty = pkl.load(f)
f.close()
print('finish loading')
def eval_osr(y_true, y_pred):
    # open-set auc-roc (binary class)

    #print(y_true.shape)
    #print(y_pred.shape)
    auroc = roc_auc_score(y_true, y_pred)

    # open-set auc-pr (binary class)
    # as an alternative, you may also use `ap = average_precision_score(labels, uncertains)`, which is approximate to aupr.
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    aupr = auc(recall, precision)

    # open-set fpr@95 (binary class)
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
    operation_idx = np.abs(tpr - 0.95).argmin()
    fpr95 = fpr[operation_idx]  # FPR when TPR at 95%
    return auroc, aupr, fpr95
#print(dicty)
joints_train_rep = np.stack(dicty['joints_train_rep'])
joints_train_out = np.stack(dicty['joints_train_out'])
train_label = np.stack(dicty['train_label'])
bones_train_rep = np.stack(dicty['bones_train_rep'])
bones_train_out = np.stack(dicty['bones_train_out'])
vels_train_rep =np.stack(dicty['vels_train_rep'])
vels_train_out = np.stack(dicty['vels_train_out'])


joints_test_seen_rep = np.stack(dicty['joints_test_seen_rep'])
joints_test_seen_out = np.stack(dicty['joints_test_seen_out'])
test_seen_label = np.stack(dicty['test_seen_label'])
bones_test_seen_rep = np.stack(dicty['bones_test_seen_rep'])
bones_test_seen_out = np.stack(dicty['bones_test_seen_out'])
vels_test_seen_rep = np.stack(dicty['vels_test_seen_rep'])
vels_test_seen_out = np.stack(dicty['vels_test_seen_out'])

joints_test_unseen_rep = np.stack(dicty['joints_test_unseen_rep'])
joints_test_unseen_out = np.stack(dicty['joints_test_unseen_out'])
test_unseen_label = np.stack(dicty['test_unseen_label'])
bones_test_unseen_rep = np.stack(dicty['bones_test_unseen_rep'])
bones_test_unseen_out = np.stack(dicty['bones_test_unseen_out'])
vels_test_unseen_rep = np.stack(dicty['vels_test_unseen_rep'])
vels_test_unseen_out = np.stack(dicty['vels_test_unseen_out'])

###########################################################
# calculate good samples
###########################################################


print('KNN joints')
neigh_joints =  NearestNeighbors(n_neighbors=3).fit(joints_train_rep)
print('KNN bones')
neigh_bones =  NearestNeighbors(n_neighbors=3).fit(bones_train_rep)
print('KNN vels')
neigh_vels =  NearestNeighbors(n_neighbors=3).fit(vels_train_rep)

dist_joints = neigh_joints.kneighbors(np.concatenate([joints_test_seen_rep, joints_test_unseen_rep],0))[0]
pred_ind = neigh_joints.kneighbors(np.concatenate([joints_test_seen_rep, joints_test_unseen_rep],0))[1][:,0]
dist_bones = neigh_bones.kneighbors(np.concatenate([bones_test_seen_rep, bones_test_unseen_rep],0))[0]
pred_ind_bones = neigh_bones.kneighbors(np.concatenate([bones_test_seen_rep, bones_test_unseen_rep],0))[1][:,0]
dist_vels = neigh_vels.kneighbors(np.concatenate([vels_test_seen_rep, vels_test_unseen_rep],0))[0]
pred_ind_vels = neigh_vels.kneighbors(np.concatenate([vels_test_seen_rep, vels_test_unseen_rep],0))[1][:,0]
dist_concat = np.stack([dist_joints[:,0], dist_bones[:,0], dist_vels[:,0]], 0)
pred_concat = np.stack([pred_ind, pred_ind_bones, pred_ind_vels], 0)
#print(dist_joints.shape)
index = np.argmin(dist_concat, axis=0)
pred_indl = []
#print(index.shape)
for ind in range(pred_concat.shape[1]):
    #print(pred_concat[index[ind], ind])
    pred_indl.append(pred_concat[index[ind], ind])
pred_indl = np.stack(pred_indl)
#print(pred_ind.shape)
#print(train_label.shape)
#print(pred_ind)
pred_labels = train_label[pred_indl[:test_seen_label.shape[0]]]
acc = 0.0
for i, item in enumerate(pred_labels):
    if item == test_seen_label[i]:
        acc += 1

probab_joints = np.max(dist_joints, -1)[np.newaxis, :][0]
#print(normalize(np.max(dist_joints, -1)[np.newaxis, :], axis=1))
probab_bones = np.max(dist_bones, -1)[np.newaxis, :][0]


probab_vels = np.max(dist_vels, -1)[np.newaxis, :][0]

###############################eval before partition of the inw and ino #####################################################
#probab_joints = np.max(neigh_joints.predict_proba(np.concatenate([joints_test_seen_rep, joints_test_unseen_rep],0)),-1)
probab_labels = np.concatenate([np.zeros(joints_test_seen_rep.shape[0]), np.ones(joints_test_unseen_rep.shape[0])])


#probab_bones = np.max(neigh_bones.predict_proba(np.concatenate([bones_test_seen_rep, bones_test_unseen_rep],0)),-1)
probab_labels = np.concatenate([np.zeros(bones_test_seen_rep.shape[0]), np.ones(bones_test_unseen_rep.shape[0])])



#probab_vels = np.max(neigh_vels.predict_proba(np.concatenate([vels_test_seen_rep, vels_test_unseen_rep],0)),-1)
#probab_vels = np.concatenate([np.zeros(vels_test_seen_rep.shape[0]), np.ones(vels_test_unseen_rep.shape[0])])



###############################eval after partition of the inw and ino #####################################################


all_dist = (probab_bones + probab_joints + probab_vels)/3

joints_pred = torch.Tensor(np.concatenate([joints_test_seen_out, joints_test_unseen_out]))
bones_pred = torch.Tensor(np.concatenate([bones_test_seen_out, bones_test_unseen_out]))
vels_pred = torch.Tensor(np.concatenate([vels_test_seen_out, vels_test_unseen_out]))
all_prob = (joints_pred + bones_pred + vels_pred)/3

l_prob = torch.max(torch.nn.functional.softmax(all_prob, -1), dim=-1)[0].numpy()
l_pred = torch.max(torch.nn.functional.softmax(all_prob, -1), dim=-1)[1].numpy()

l_prob_joints_recal = []
l_prob_bones_recal = []
l_prob_vels_recal = []

for ind in range(joints_pred.shape[0]):
    item_joints = joints_pred[ind]
    item_bones = bones_pred[ind]
    item_vels = vels_pred[ind]
    pos = l_pred[ind]
    dist = all_dist[ind]
    mask = torch.ones(40)
    mask[pos] = 0
    mask = mask.bool()
    #print(mask)


    j_upper = torch.sum(torch.exp(item_joints[mask]* dist**2 )) * (1-dist)
    j_unter = dist
    item_joints[pos] = torch.log(j_upper / j_unter)
    
    
    #print(item_joints[pos])
    item_joints[mask] = item_joints[mask] * dist**2
    
    l_prob_joints_recal.append(item_joints)
    b_upper = torch.sum(torch.exp(item_bones[mask]* dist**2))* (1-dist)
    b_unter = dist
    item_bones[pos] = torch.log(b_upper / b_unter)
    item_bones[mask] = item_bones[mask] * dist**2
    l_prob_bones_recal.append(item_bones)
    v_upper = torch.sum(torch.exp(item_vels[mask]* dist**2))* (1-dist)
    v_unter = dist
    item_vels[pos] = torch.log(v_upper / v_unter)
    item_vels[mask] = item_vels[mask] * dist**2
    l_prob_vels_recal.append(item_vels)

l_prob_joints_recal = torch.stack(l_prob_joints_recal)
l_prob_bones_recal = torch.stack(l_prob_bones_recal)
l_prob_vels_recal = torch.stack(l_prob_vels_recal)
prob_j, pred_j = torch.max(torch.nn.functional.softmax(l_prob_joints_recal, dim=-1), dim=-1)
prob_b, pred_b = torch.max(torch.nn.functional.softmax(l_prob_bones_recal, dim=-1), dim=-1)
prob_v, pred_v = torch.max(torch.nn.functional.softmax(l_prob_vels_recal, dim=-1), dim=-1)
l_prob = (l_prob_bones_recal + l_prob_vels_recal + l_prob_joints_recal)/3
prob, pred = torch.max(torch.nn.functional.softmax(l_prob, dim=-1), dim=-1)

print(eval_osr(probab_labels, 1-prob))
acc = 0.0
for i, item in enumerate(pred[:test_seen_label.shape[0]]):
    if item == test_seen_label[i]:
        acc += 1
print('ACC is: ', acc/pred_labels.shape[0])
