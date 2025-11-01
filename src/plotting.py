import csv
import matplotlib.pyplot as plt

# Print results to a csv file with no headings.
def print_results(results, fileName):
    with open(fileName, 'w') as f:
        wr = csv.writer(f)
        wr.writerows(results)


# Read results fromn a csv file with no headings.
def read_results(fileName):
    with open(fileName, 'r') as f:
        results = list(csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC))
    return results


# Formats rersults array into matplotlib frendly format.
def sort_results_for_plot(results):
    epoch = [0]
    accuracy = [0]
    precision = [0]
    recall = [0]
    f1 = [0]
    auc = [0]
    timetaken = [0]
    for i in results:
        epoch.append(i[0])
        accuracy.append(i[1])
        precision.append(i[2])
        recall.append(i[3])
        f1.append(i[4])
        auc.append(i[5])
        timetaken.append(i[6])
    
    return [epoch, accuracy, precision, recall, f1, auc, timetaken]


# Makes two plots. The first shows all the measure in one figure. The second shows the time taken.
def one_method_plot(plotResults, method):
    if method == "VQC":
        xlable = "Tolerance"
    else: 
        xlable = "Epochs"
        
    plt.figure(1)
    plt.plot(plotResults[0], plotResults[1], label = 'Accuracy')
    plt.plot(plotResults[0], plotResults[2], '.-', label = 'Precision')
    plt.plot(plotResults[0], plotResults[3], '--', label = 'Recall')
    plt.plot(plotResults[0], plotResults[4], '*-', label = 'F1')
    plt.plot(plotResults[0], plotResults[5], 'o-', label = 'AUC')
    #plt.ylim(0,1)
    plt.xlabel(xlable)
    plt.ylabel("Probablity")
    plt.legend()
    titleMeas = method + " Measures"
    plt.title(titleMeas)
    plt.savefig(titleMeas)
    plt.close()
    
    plt.figure(2)
    plt.plot(plotResults[0], plotResults[6])
    plt.xlabel(xlable)
    plt.ylabel("Time Taken (s)")
    titleTime = method + " Time"
    plt.title(titleTime)
    plt.savefig(titleTime)
    plt.close()


# Makes a plot of one of the measurs or time taken for two different methods.
def one_measure_plot(plotResults1, plotResults2, measure, method1, method2):
    measureType = {1:'Accuracy', 2:'Precision', 3:'Recall', 4:'F1', 5:'AUC', 6:'Time(s)'}
    plt.figure(1)
    plt.plot(plotResults1[0], plotResults1[measure], label = method1)
    plt.plot(plotResults2[0], plotResults2[measure], '.-', label = method2)
    plt.xlabel("Epochs")
    plt.ylabel(measureType[measure])
    plt.legend()
    titleMeas = method1 + " " + method2 + " " + measureType[measure]
    plt.title(titleMeas)
    plt.savefig(titleMeas)
    plt.close()