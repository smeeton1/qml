import time
import classical_analysis as cl
import quantum_analysis as qu

# function to generate various preformance paramiters for classical ai
def generate_classical_results(maxEpochs, steps, tol, fileName, para, targetName):
    trainLoader, testLoader, paramiterTrain, targetTest=cl.read_prepare_data(fileName, para, targetName)
    inputDim = paramiterTrain.shape[1] 
    results = []

    for i in range(steps,maxEpochs,steps):
        print('-----------------------------------------')
        print('Epochs: ', i)
        model = cl.MediNN(inputDim)
        start = time.time()
        model = cl.train_model(model, trainLoader, i, tol)
        end = time.time()
        timetaken = end - start
        accuracy, precision, recall, f1, auc = cl.test_model(model, testLoader, targetTest)
        results.append([i, accuracy, precision, recall, f1, auc, timetaken])

    return results



def generate_vqc_results(fileName, targetName):
    paramiterTrain, paramiterTest, targetTrain, targetTest = qu.qml_read_prepare_data(fileName, targetName)
    results = []

    for i in [1e-1]:
        print('-----------------------------------------')
        print('Tolerance: ', i)
        vqc = qu.create_vqc(paramiterTrain.shape[1], 3, i)
        start = time.time()
        vqc.fit(paramiterTrain, targetTrain)
        end = time.time()
        timetaken = end - start
        accuracy, precision, recall, f1, auc = qu.test_vqc(vqc, paramiterTest, targetTest)
        results.append([i, accuracy, precision, recall, f1, auc, timetaken])

    return results

def generate_qnn_results(maxEpochs, steps, tol,fileName, targetName):
    paramiterTrain, paramiterTest, targetTrain, targetTest = qu.qml_read_prepare_data(fileName, targetName)
    inputDim = paramiterTrain.shape[1]
    results = []
    

    for i in range(steps,maxEpochs,steps):
        print('-----------------------------------------')
        print('Epochs: ', i)
        qnn = qu.create_qnn(inputDim, 1)
        model = qu.creat_pytorch_qnn(qnn)
        start = time.time()
        model = qu.train_qnn(model, paramiterTrain, targetTrain, i, tol)
        end = time.time()
        timetaken = end - start
        accuracy, precision, recall, f1, auc = qu.test_qnn(model, paramiterTest, targetTest)
        results.append([i, accuracy, precision, recall, f1, auc, timetaken])
    
    return results