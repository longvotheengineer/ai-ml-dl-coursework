from __future__ import print_function
import matplotlib, sys
from matplotlib import pyplot as plt
import numpy as np

def predict(inputs,weights):
    activation=0.0
    for i,w in zip(inputs,weights):
        activation += i*w 
    return 1.0 if activation>=0.0 else 0.0

def plot(matrix,weights=None,title="Prediction Matrix"):
    if len(matrix[0])==5:  # 3D inputs (bias + 3 features + y)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(title)
        ax.set_xlabel("i1")
        ax.set_ylabel("i2")
        ax.set_zlabel("i3")

        # Decision boundary plane
        if weights is not None:
            x = np.linspace(0,1,10)
            y = np.linspace(0,1,10)
            xx,yy = np.meshgrid(x,y)
            # w0 + w1*x + w2*y + w3*z = 0  -> solve for z
            zz = -(weights[0] + weights[1]*xx + weights[2]*yy) / weights[3]
            ax.plot_surface(xx,yy,zz,alpha=0.3,color="green")

        c1_data=[[],[],[]]
        c0_data=[[],[],[]]
        for i in range(len(matrix)):
            cur_i1 = matrix[i][1]
            cur_i2 = matrix[i][2]
            cur_i3 = matrix[i][3]
            cur_y  = matrix[i][-1]
            if cur_y==1:
                c1_data[0].append(cur_i1)
                c1_data[1].append(cur_i2)
                c1_data[2].append(cur_i3)
            else:
                c0_data[0].append(cur_i1)
                c0_data[1].append(cur_i2)
                c0_data[2].append(cur_i3)

        ax.scatter(c0_data[0],c0_data[1],c0_data[2],s=40.0,c='r',label='Class 0')
        ax.scatter(c1_data[0],c1_data[1],c1_data[2],s=40.0,c='b',label='Class 1')

        plt.legend(fontsize=10,loc=1)
        plt.show()
        return

    print("Matrix dimensions not covered.")

def accuracy(matrix,weights):
    num_correct = 0.0
    preds       = []
    for i in range(len(matrix)):
        pred   = predict(matrix[i][:-1],weights) 
        preds.append(pred)
        if pred==matrix[i][-1]: num_correct+=1.0 
    print("Predictions:",preds)
    return num_correct/float(len(matrix))

def train_weights(matrix,weights,nb_epoch=10,l_rate=1.00,do_plot=False,stop_early=True,verbose=True):
    for epoch in range(nb_epoch):
        cur_acc = accuracy(matrix,weights)
        print("\nEpoch %d \nWeights: "%epoch,weights)
        print("Accuracy: ",cur_acc)
        
        if cur_acc==1.0 and stop_early: break 
        if do_plot: plot(matrix,weights,title="Epoch %d"%epoch)
        
        for i in range(len(matrix)):
            prediction = predict(matrix[i][:-1],weights) 
            error      = matrix[i][-1]-prediction		 
            if verbose: sys.stdout.write("Training on data at index %d...\n"%(i))
            for j in range(len(weights)): 				 
                if verbose: sys.stdout.write("\tWeight[%d]: %0.5f --> "%(j,weights[j]))
                weights[j] = weights[j]+(l_rate*error*matrix[i][j]) 
                if verbose: sys.stdout.write("%0.5f\n"%(weights[j]))

    plot(matrix,weights,title="Final Epoch")
    return weights 

def main():
    nb_epoch        = 20
    l_rate          = 1.0
    plot_each_epoch = False
    stop_early      = True

    # Example 3D dataset (bias + 3 inputs + label)
    matrix = [
        [1.0, 0.1, 0.2, 0.7, 1.0],
        [1.0, 0.2, 0.8, 0.3, 0.0],
        [1.0, 0.3, 0.3, 0.9, 1.0],
        [1.0, 0.7, 0.6, 0.2, 0.0],
        [1.0, 0.9, 0.1, 0.8, 1.0],
        [1.0, 0.8, 0.7, 0.4, 0.0],
    ]
    weights = [0.2, 1.0, -1.0, 0.5] # initial weights

    train_weights(matrix,weights=weights,nb_epoch=nb_epoch,l_rate=l_rate,
                  do_plot=plot_each_epoch,stop_early=stop_early)

if __name__ == '__main__':
    main()
