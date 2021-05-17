import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
import os


Shape = 887

base_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'train_job')

## load        
job_companies_dir = os.path.join(base_dir, 'job_companies.csv')
job_companies = pd.read_csv(job_companies_dir)

job_tags_dir = os.path.join(base_dir, 'job_tags.csv')
job_tags = pd.read_csv(job_tags_dir)

tags_dir = os.path.join(base_dir, 'tags.csv')
tags = pd.read_csv(tags_dir)

train_dir = os.path.join(base_dir, 'train.csv')
train = pd.read_csv(train_dir)

user_tags_dir = os.path.join(base_dir, 'user_tags.csv')
user_tags = pd.read_csv(user_tags_dir)

test_job_dir = os.path.join(base_dir, 'test_job.csv')
test_job = pd.read_csv(test_job_dir)

DicTagIDtoIdx = {}
for idx, row in tags.iterrows():
        DicTagIDtoIdx[row['tagID']] = idx

        

def getTagArrayFromUserID(userIDtarget):
        ListToArray = [0]*887
        for _, row in user_tags.iterrows():
                if row['userID'] == userIDtarget:
                        idx = DicTagIDtoIdx[row['tagID']]
                        ListToArray[idx] = 1

        return ListToArray
        # a = np.array(ListToArray).reshape(-1, 1)
        # print(a)

def getTagArrayFromJobID(jobIDtarget):
        ListToArray = [0]*887
        for _, row in job_tags.iterrows():
                if row['jobID'] == jobIDtarget:
                        idx = DicTagIDtoIdx[row['tagID']]
                        ListToArray[idx] = 1

        return ListToArray


def convertData(dataFrame):
        resultArray = []

        for idx, row in dataFrame.iterrows():
                userIDTagArray = getTagArrayFromUserID(row['userID'])
                jobIDTagArray = getTagArrayFromJobID(row['jobID'])

                new_list = list(map(lambda x, y : x*y, userIDTagArray, jobIDTagArray))

                resultArray.append(new_list)
                print("idx : {} completed".format(idx))

        return resultArray

        
## for generating train_result
# train_X = np.array(convertData(train))
# train_y = train['applied'].to_numpy()
# pd.DataFrame(train_X).to_csv("train_result.csv")

## for generating test_result
# test_X = np.array(convertData(test_job))
# pd.DataFrame(test_X).to_csv("test_result.csv")


## load train
train_X = pd.read_csv('train_result.csv')
train_X = train_X.drop(train_X.columns[0], axis=1)
train_y = train['applied'].to_numpy().astype(np.uint8)

## load test_X
test_X = pd.read_csv('test_result.csv')
test_X = test_X.drop(test_X.columns[0], axis=1)
print(train_X.shape)

## set weight
neg, pos = np.bincount(train_y)
total = neg + pos
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))

## class weight
weight_for_0 = (1 / neg)*(total)/2.0 
weight_for_1 = (1 / pos)*(total)/2.0

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))


## model
model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(887,)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
]) 

model.compile(
      optimizer=tf.keras.optimizers.RMSprop(lr=1e-3),\
      loss=tf.keras.losses.BinaryCrossentropy(),\
      metrics=['accuracy'])


model.fit(train_X, train_y, epochs=20, class_weight=class_weight)


# test_loss, test_acc = model.evaluate(eval_X,  eval_y, verbose=2)

# print('\n테스트 정확도:', test_acc)


# predictions = model.predict_classes(test_X)
predictions = (model.predict(test_X) > 0.5).astype("int32")
pd.DataFrame(predictions, columns=['applied']).to_csv("result.csv", index=None)


predictions = model.predict(test_X)
print(predictions)