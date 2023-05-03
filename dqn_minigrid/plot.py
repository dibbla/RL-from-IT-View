# Read the information in given txt files and plot the results for comparison

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# plot the same env results for 100% memorization
def plot_same_env_results():
    file1_path = "logs/swipe-log-20230502-132827hidden-16/returns.txt"
    file2_path = "logs/swipe-log-20230502-133244hidden-32/returns.txt"
    file3_path = "logs/swipe-log-20230502-133500hidden-64/returns.txt"
    file4_path = "logs/swipe-log-20230502-134252hidden-72/returns.txt"
    file5_path = "logs/swipe-log-20230502-135030hidden-128/returns.txt"

    file1 = open(file1_path, "r")
    file2 = open(file2_path, "r")
    file3 = open(file3_path, "r")
    file4 = open(file4_path, "r")
    file5 = open(file5_path, "r")

    # read the data
    data1 = file1.readlines()
    data2 = file2.readlines()
    data3 = file3.readlines()
    data4 = file4.readlines()
    data5 = file5.readlines()

    print(len(data1),len(data2),len(data3),len(data4),len(data5))
    # smooth the data
    for i in range(len(data1)):
        data1[i] = float(data1[i])
        data2[i] = float(data2[i])
        data3[i] = float(data3[i])
        data4[i] = float(data4[i])
        data5[i] = float(data5[i])
    
    # use moving average with 10 neighbours to smooth the data with np.sum
    for i in range(len(data1)-10):
        data1[i] = np.sum(data1[i:i+10])/10
        data2[i] = np.sum(data2[i:i+10])/10
        data3[i] = np.sum(data3[i:i+10])/10
        data4[i] = np.sum(data4[i:i+10])/10
        data5[i] = np.sum(data5[i:i+10])/10

    # close the files
    file1.close()
    file2.close()
    file3.close()
    file4.close()
    file5.close()

    # plot the data
    plt.plot(data1, label="hidden-16")
    plt.plot(data2, label="hidden-32")
    plt.plot(data3, label="hidden-64")
    plt.plot(data4, label="hidden-72")
    plt.plot(data5, label="hidden-128")

    # define y axis
    plt.ylim(0, 1.1)

    plt.xlabel("Episodes")
    plt.ylabel("Return")
    plt.title("100% Memorization")
    plt.legend()
    # save the plot
    plt.savefig("100%_memorization.png")

# plot_same_env_results()

# plot the same hidden layer results for generalization
def plot_generalization_results():
    file1_path = "logs/swipe-log-20230502-132827hidden-16/eval_returns.txt"
    file2_path = "logs/swipe-log-20230502-133244hidden-32/eval_returns.txt"
    file3_path = "logs/swipe-log-20230502-133500hidden-64/eval_returns.txt"
    file4_path = "logs/swipe-log-20230502-134252hidden-72/eval_returns.txt"
    file5_path = "logs/swipe-log-20230502-135030hidden-128/eval_returns.txt"
    file6_path = "logs/swipe-log-20230503-140106hidden-28/eval_returns.txt"

    file1 = open(file1_path, "r")
    file2 = open(file2_path, "r")
    file3 = open(file3_path, "r")
    file4 = open(file4_path, "r")
    file5 = open(file5_path, "r")
    file6 = open(file6_path, "r")

    # read the data
    data1 = file1.readlines()
    data2 = file2.readlines()
    data3 = file3.readlines()
    data4 = file4.readlines()
    data5 = file5.readlines()
    data6 = file6.readlines()

    # get the max of data
    max_data1 = max(data1)
    max_data2 = max(data2)
    max_data3 = max(data3)
    max_data4 = max(data4)
    max_data5 = max(data5)
    max_data6 = max(data6)
    
    # convert to float
    max_data1 = float(max_data1)
    max_data2 = float(max_data2)
    max_data3 = float(max_data3)
    max_data4 = float(max_data4)
    max_data5 = float(max_data5)
    max_data6 = float(max_data6)


    # set x axis to be 16, 28, 32, 64, 72, 128
    x = [16, 28, 32, 64, 72, 128]
    y = [max_data1, max_data6, max_data2, max_data3, max_data4, max_data5]

    print(y)

    # close the files
    file1.close()
    file2.close()
    file3.close()
    file4.close()
    file5.close()
    file6.close()

    # plot the data, in scatter plot
    plt.scatter(x, y)
    plt.plot(x, y)
    # define labels
    plt.xlabel("Hidden Layer Size")
    plt.ylabel("Return")

    # define x axis
    plt.savefig("generalization.png")

plot_generalization_results()