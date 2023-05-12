import matplotlib.pyplot as plt

# # draw capacity progression figure
# x = [5,10,20,40,80,100]
# y = [9,10,11,11,12,12]
# plt.plot(x,y)

# # set y axis range
# plt.ylim(7,13)

# plt.xlabel('% Training Data')
# plt.ylabel('MEC % Overfitting for 100% Accuracy')
# plt.title('Capacity progression curve for network finding a rule')
# # plt.savefig('capacity_progression.png')

# draw MEC data figure
MEC = [20, 25, 30, 36, 42, 48, 53, 59, 63, 69, 74, 77]
Acc_on_Env = [0.2, 0.3640, 0.4880, 0.6, 0.6, 0.8, 0.9960, 1,1,1,1,1]
Acc_on_D1 = [0.2340, 0.2070,0.3410,0.4780,0.23,0.6020,0.797,0.979,1,0.8240,0.8490,0.7620]
Acc_on_D2 = [0.2690,0.1860,0.306,0.4320,0.1740,0.5680,0.6110,0.6500,0.7670,0.6350,0.7520,0.5220]
print(len(MEC), len(Acc_on_Env), len(Acc_on_D1), len(Acc_on_D2) )
plt.plot(MEC, Acc_on_Env, label='Accuracy on Environment')
plt.plot(MEC, Acc_on_D1, label='Accuracy on D1')
plt.plot(MEC, Acc_on_D2, label='Accuracy on D2')
plt.legend()
plt.xlabel('MEC')
plt.ylabel('Validation Accuracy')
plt.title('Accuracy on Environment, Drift 1 and Drift 2')
plt.savefig('MEC.png')
