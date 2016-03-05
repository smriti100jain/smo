import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import numpy as np
from operator import add
import math
import time



test=0
def kernel(i1,i2):
    kernel_value=0.
    if(test==0):
	#guassian kernel
        for i in range(len(training_set[i1])):
            kernel_value=kernel_value+(training_set[i1][i]-training_set[i2][i])**2
        kernel_value=-0.5*kernel_value
        return (math.e**kernel_value)
	#linear kernel
    for i in range(len(training_set[i1])):
        kernel_value=kernel_value+training_set[i1][i]*training_set[i2][i]
    return kernel_value

def SVM_OUTPUT(i2):
    
    global target
    global threshold
    svm_out=0

    for i in range(len(target)):
        svm_out=svm_out+alpha[i]*target[i]*kernel(i,i2)
    svm_out=svm_out-threshold


    return svm_out

def takeStep(i1,i2):
    global threshold
    global target
    global training_set
    #initialize eps
    if (i1==i2):
        return 0
    alph1=alpha[i1]
    y1=target[i1]
    y2=target[i2]
    E1=SVM_OUTPUT(i1)-y1 
    E2=SVM_OUTPUT(i2)-y2
    
    s=y1*y2
    alph2=alpha[i2]
    if(not(y1==y2)):
        L=max(0,(alph2-alph1))
        H=min(C,(C+alph2-alph1))
    else:
        L=max(0,(alph2+alph1-C))
        H=min(C,(alph2+alph1))
    if(L==H):
        return 0
    k11=kernel(i1,i1)
    k12=kernel(i1,i2)
    k22=kernel(i2,i2)
    eta=2*k12-k11-k22
    if(eta<0):
        a2=alph2-y2*(E1-E2)/eta
        if(a2<L):
            a2=L
        elif(a2>H):
            a2=H
    else:
        temp_coeff=alpha[:]
        temp_coeff[i2]=L
        t1=0
        t2=0
        for i in range(len(alpha)):
            t1+=temp_coeff[i]
        for i in range(len(alpha)):
            for j in range(len(alpha)):
                t2+=target[i]*target[j]*kernel(i,j)*temp_coeff[i]*temp_coeff[j]
        Lobj=t1-1/2*t2

        temp_coeff=alpha[:]
        temp_coeff[i2]=H
        t1=0
        t2=0
        for i in range(len(alpha)):
            t1+=temp_coeff[i]
        for i in range(len(alpha)):
            for j in range(len(alpha)):
                t2+=target[i]*target[j]*kernel(i,j)*temp_coeff[i]*temp_coeff[j]
        Hobj=t1-1/2*t2
        if (Lobj>Hobj+eps):
            a2=L
        elif(Lobj<Hobj-eps):
            a2=H
        else:
            a2=alph2
    

    if(a2 < 0.00000001):
        a2=0
    elif(a2 > C - 0.00000001):
        a2=C
    if(abs(a2-alph2)<eps*(a2+alph2+eps)):
        return 0
        
    a1=alph1+s*(alph2-a2)
        
    b1=E1+threshold+y1*(a1-alph1)*k11+y2*(a2-alph2)*k12
    b2=E2+threshold+y1*(a1-alph1)*k12+y2*(a2-alph2)*k22
    t=-1
    bold=threshold
    if(not(a1==0 or a1==C)):
        threshold=b1
    elif(not(a2==0 or a2==C)):
        threshold=b2
    else:
        threshold=(b1+b2)/2
    

    arr=[i for i in range(len(error))]
    

    if(not(t==-1)):
        error[t]=0

    arr.remove(i1)
    arr.remove(i2)
    for i in arr:
        temp=error[i]+y1*(a1-alph1)*kernel(i1,i)+y2*(a2-alph2)*kernel(i2,i)+bold-threshold
        error[i]=temp
        temp=0
    
    alpha[i1]=a1
    alpha[i2]=a2

    return 1
        

            
def update_plane(num):
    global mesh_plot
    ax.collections.remove(mesh_plot)
    mesh_plot=ax.plot_surface(meshgrid_list[num][0],meshgrid_list[num][1],meshgrid_list[num][2], color='green',alpha=0.5)
    return mesh_plot

            
def update_plane_kernel(num):
    global mesh_plot
    global points
    global norm
    for i in points:
        i.remove()
    points=[]
    for i in range(len(training_set)):
        if(norm[num][i]>0):
            points.append(ax.scatter(training_set[i][0], training_set[i][1], training_set[i][2], c='r', marker='o'))
        else:
            points.append(ax.scatter(training_set[i][0], training_set[i][1], training_set[i][2], c='b', marker='o'))


    return points
        
    
    
    
def examineExample(i2):
    y2=target[i2]
    alph2=alpha[i2]  #check
    E2=SVM_OUTPUT(i2) - y2 
    r2=E2*y2
    
    
    if((r2<-tol and alph2<C) or (r2>tol and alph2>0)):
        for i in range(len(alpha)):
            if (not(alpha[i]==0) and not(alpha[i]==C)):
                if takeStep(i,i2):
                    return 1
        
        for i in range(len(alpha)):
            if takeStep(i,i2):
                return 1
        
    return 0



#-------------------------------------------------------------------------------
#training_set
#target
global norm
norm=[]
eps=0.001
tol=0.001
C=1
threshold=0
temp=0

file1=1
if (file1==1):

    f=open('input2.txt')
    training=[]
    target=[]

    for line in f:
        temp=line.split()
        temp=[int(i) for i in temp]
        training.append(temp)
    print training
    print "kkk"
    print len(training[0])
    training_set=[[0 for i in range(len(training[0])-1)] for j in range(len(training))]
    print len(training_set)
    print len(training_set[0])
    for i in range(len(training)):
        for j in range(len(training[0])-1):
            training_set[i][j]=training[i][j+1]
        temp=training[i][0]
        print temp
        if (temp == 2 or temp == 1):
            temp=-1
        elif(temp == 0):
            temp=1
        print temp
        target.append(temp)

elif(file1==2):
    f=open('input1.txt')
    training=[]
    target=[]
    global training_set
    training_set=[]
    for line in f:
        temp=line.split(',')
        temp=[int(i) for i in temp]
        training.append(temp)
    print len(training)
    print len(training[0])
    training_set=[[0 for i in range(len(training[0])-1)] for j in range(len(training))]
    print len(training_set)
    print len(training_set[0])
    for i in range(len(training)):
        for j in range(len(training[0])-1):
            training_set[i][j]=training[i][j]
        temp=training[i][len(training[0])-1]
        if (temp==2):
            temp=-1
        target.append(temp)

elif(file1==3):
    global training_set
    training_set=[]
    global target
    target=[]
    f=open('input.txt')
    training=[]
    target=[]

    for line in f:
        temp=line.split(',')
        temp=[int(i) for i in temp]
        training_set.append(temp)

    f=open('target.txt')
    for line in f:
        temp=int(line)
        target.append(temp)


print training_set
print target




k=len(training_set)
alpha=[0 for i in range(k)]
error=[0 for i in range(k)]

threshold=0.
numChanged=0
examineAll=1
normal_list=[]
point_list=[]
threshold_list=[]
while(numChanged>0 or examineAll):
    numChanged=0
    print "goo"
    if(examineAll):
       
        for i in range(len(training_set)):
            numChanged=numChanged+examineExample(i)

                
                    


            
    else:
       
        for i in range(len(training_set)):
            if(not(alpha[i]==0) and not(alpha[i]==C)):
                numChanged=numChanged+examineExample(i)
            
    if(examineAll==1):
        examineAll=0
    elif(numChanged==0):
        examineAll=1

    index=0
    
    W=[0 for iter2 in range(len(training_set[0]))]
    for iter2 in range(len(alpha)):
        if alpha[iter2]>0:
            temp=[target[iter2]*alpha[iter2]*iter3 for iter3 in training_set[iter2]]
            #print temp,coeff_list[iter2]
            W=map(add,temp,W)

    for iter2 in range(len(W)):
        if W[iter2]>0:
            index=iter2
            break

    pt=[]
    for iter2 in range(len(training_set[0])):
        if index==iter2:
            pt.append(threshold/W[iter2])
        else:
            pt.append(0)

    normal=np.array(W)
    point=np.array(pt)

    point_list.append(point)
    normal_list.append(normal)
    threshold_list.append(threshold)
    temp=[]
    global norm
    for u in range(len(training_set)):
        temp.append(SVM_OUTPUT(u))

    norm.append(temp)
global points
points=[]

meshgrid_list=[]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for plot_data in range(len(training_set)):
    if target[plot_data]==-1:           
        points.append(ax.scatter(training_set[plot_data][0], training_set[plot_data][1], training_set[plot_data][2], c='r', marker='o'))
    elif target[plot_data]==1:          
        points.append(ax.scatter(training_set[plot_data][0], training_set[plot_data][1], training_set[plot_data][2], c='b', marker='o'))


for iter_plot in range(len(normal_list)):
    normal=normal_list[iter_plot]
    threshold=threshold_list[iter_plot]
    xx, yy = np.meshgrid(range(100), range(100))
    z1 = (-normal[0]*xx - normal[1]*yy + threshold)*1./normal[2]
    meshgrid_list.append([xx,yy,z1])

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')    
ax.set_zlabel('Z Label')
if(test==1):
    mesh_plot=ax.plot_surface(meshgrid_list[0][0],meshgrid_list[0][1],meshgrid_list[0][2], color='yellow')
if(test==1):
    line_ani = animation.FuncAnimation(fig, update_plane, len(normal_list),
                              interval=500, blit=False)
else:
    line_ani = animation.FuncAnimation(fig, update_plane_kernel, len(normal_list),
                              interval=1000, blit=False)

plt.show()



