"""
Univariate LR / Single Variable / Single feature
1) For the data in the attached file (univariate_linear_regression.csv), 
(a) do the linear regression (best line fit) 
using steepest gradient descent with line (univariate) search. 

(b) Plot the best fit line, cost function 
as well as the contour plot of the cost function.
"""

"""
Please note:
Part a is implemenented in this .py file

Part b is implemented in attached .ipynb file
and all plots are verified to be running in Google Colab
"""

import pandas as pd
import math 

#SOLVING FOR PART a of Q1

#1) For the data in the attached file (univariate_linear_regression.csv), 
#(a) do the linear regression (best line fit) 
#using steepest gradient descent with line (univariate) search. 

	
# header names
header = ['x', 'y']
	
# define name of csv file
filename = "univariate_linear_regression.csv"

#Read csv file Using Pandas
df = pd.read_csv(filename)
#print(df)

x_data = [] #list to store x column data from csv file
y_data = [] #list to store y column data from csv file

#read data from csv file and store in corresponding lists
for ind in df.index: #get index value from data frame
    #print data row wise
    #print(ind, df[header[0]][ind], df[header[1]][ind]) 
    x_data.append(df[header[0]][ind])
    y_data.append(df[header[1]][ind])
print("\n")

# m = number of sample points available in ground truth data
m = len(x_data) 
#print(m)





############################################################### 1
"""
Setting up functions for finding optimal value of alpha along grad J 
"""
#FUNCTION 1 
#to compute J for given value of alpha during bracketing method
def J_for_w_bracketing(alpha, j_grad_w1, j_grad_w2, i=0):
    
    #compute w_0 and w_1 using alpha as single variable to be optimised
    #along grad J
    #any starting point still leads to convergence
    w_0 = -5 + (alpha * j_grad_w1) #w_0 in terms of alpha
    w_1 = 5 + (alpha * j_grad_w2) #w_1 in terms of alpha

    #return value of w_0, w_1, J 
    if i == 1:
        return w_0, w_1, (w_0 + (w_1 * x))    
    
    #return value of J
    else:
        return (w_0 + (w_1 * x))


#FUNCTION 2
#to obtain bracketed values of alpha using bracketing algorithm OR
#to obtain coarse values of alphas 
def alpha_bracketing(w1,w2, j_grad_w1, j_grad_w2):
    #here w_0 = w1 and w_1 = w2
    
    #(a) bracketing method (choose your own a, b, n).
    #initialise variables
    n = 100  # no. of iterations or step size
    a = 0.01 # initial lower boundary point guess for alpha
    b = 0.05 # initial upper boundary point guess for alpha
    
    #Step 1:Initialise variables 
    delta_alpha = (b-a)/n   #step increment for alpha 
    alpha_1 = a
    alpha_2 = alpha_1 + delta_alpha
    alpha_3 = alpha_2 + delta_alpha
    
    #Step 2: compare J(w) values based on alpha values along grad J 
    for i in range(n):    
        if J_for_w_bracketing(alpha_1,j_grad_w1, j_grad_w2) >= J_for_w_bracketing(alpha_2,j_grad_w1, j_grad_w2) and J_for_w_bracketing(alpha_2,j_grad_w1, j_grad_w2) <= J_for_w_bracketing(alpha_3, j_grad_w1, j_grad_w2):
            #print("The required values of a,b are:", w1, w3)
            break #when boundary points for alpha found, break from loop
        else: #update points alpha_1,alpha_2,alpha_3 using prev. values and delta_alpha 
            alpha_1 = alpha_2
            alpha_2 = alpha_3 
            alpha_3 = alpha_2 + delta_alpha
            if alpha_3 <= b: #if alpha_3 is within boundary, continue from step 2
                continue
            else: #else break from loop
                #print("No optimal alpha value exists in this interval or check for bdry values")
                break 
        
    #Result: output from bracketing method 
    a = alpha_1 #left boundary point for alpha 
    b = alpha_3 #right boundary point for alpha 
    #print("Output from bracketing method is:\nmin alpha = ",a, "\nmax alpha = ",b)
    
    return a, b


#FUNCTION 3
#to obtain optimal value of alpha using interval halving algorithm after bracketing method 
def alpha_interval_halving(a,b, j_grad_w1, j_grad_w2):
    
    #(b) Use the bracketed value to get to the optimal value of alpha along grad J   
    #employing interval halving method  
    stop_value = 0.000001 # define limit of search 
    am = (a+b) / 2 #start with mid value using obtained range for alpha 
    L = (a - b)/2  #define scope of boundary search region 
    
    j_am = J_for_w_bracketing(am, j_grad_w1, j_grad_w2) # compute initial value of J for mid value of alpha range
    
    for i in range(1000): #define no. of iterations for search
        #STEP 2: Set intermediate values of alpha (a1 and a2) and compute respective J values 
        a1 = a + (L/4)
        a2 = b - (L/4)   
        j_a1 = J_for_w_bracketing(a1, j_grad_w1, j_grad_w2)
        j_a2 = J_for_w_bracketing(a2, j_grad_w1, j_grad_w2)
        
        #STEP 3: check for condition of region elimination
        if j_a1 < j_am: #update values and proceed to step 5
            b = am
            am = a1
            #go to step 5
        else:
            #STEP 4: check for condition of region elimination
            if j_a2 < j_am: #update values and proceed to step 5
                a = am
                am = a2 # or b ?
            else: #update values and proceed to step 5
                a = a1
                b = a2
        #STEP 5:check if limit of search has reached
        L = b-a
        if abs(L) < stop_value:
            #print("Required critical value is: am = ", am)
            break  #break from loop 
        else:
            continue #go to step 2 and continue with next iterations
    
    #print("\n")
    #print("Critical point from interval halving method is:\nalpha = ",am)
    
    #critical_1, critical_2, j_at_critical = J_for_w(am,1)
    #print("\n")
    #print("Optimal points w1 and w2 of J(w1,w2) are:", critical_1, critical_2)
    #print("\nMinimum value of J(w1,w2) at optimal value of alpha is: ", j_at_critical)
    
    return am #optimal value of alpha
############################################################### 1
    





##################################################### 2
"""
Setting up functions for finding grad J in Gradiend Descent Algorithm 
"""
#FUNCTION 4
#to compute J(w_0,w_1) 
def J_for_w(w_0, w_1, x, i=0):
    
    #return value of w_0, w_1, J
    if i == 1:
        return w_0, w_1, (w_0 + (w_1 * x))    
    
    #return value of J
    else:
        return (w_0 + (w_1 * x))


#FUNCTION 5
#to compute first order partial derivative of J wrt w_0 at w_0, w_1 using numerical finite difference method
def J_grad_w1(w_0,w_1):
    sum_0 = 0
    
    for i in range(0,m):
        sum_0 = sum_0 + (J_for_w(w_0, w_1, x_data[i]) - y_data[i])
    
    sum_0 = sum_0 / m             
    return sum_0

#FUNCTION 6
#to compute first order partial derivative of J wrt w_1 at w_0, w_1 using numerical finite difference method
def J_grad_w2(w_0,w_1):
    sum_1 = 0
    
    for i in range(0,m):
        sum_1 = sum_1 + ((J_for_w(w_0, w_1, x_data[i]) - y_data[i]) * x_data[i])
    
    sum_1 = sum_1 / m             
    return sum_1
##################################################### 2





"""
START OF EXECUTION OF FULL ALGORITHM ---------------
"""
############################## 3   
#START OF ALGORITHM : 
#LINEAR REGRESSION USING STEEPEST GRADIENT DESCENT USING ALPHA UPDATION AT EACH STEP (VIA LINE SEARCH)      


#Hypothesis equation for univariate linear regression
#h(w_0,w_1,x) = w_0 + w_1 * x
#w_0 = bias or weight = y intercept of line fit
#w_1 = weight = slope of line fit

#cost function for univariate linear regression
#J(w_0, w_1) = sum(1 to m) (1/(2*m)) (w_0 + w_1 * x_data - y_data)^2
#x_data = ground truth x data
#y_data = ground truth y data
#m = number of available sample points


#STEP 1 : Choose initial values
#initial guess for critical point, any starting point still leads to convergence
w_0 = -5 #initial weight (bias) value  
w_1 = 5  #initial weight (slope) value
x = 0    #initial x_data value 

stop_value = 0.0001 #define limit of search or convergence for norm of gradient
k = 1               #iteration counter 
iteration = 1000    #set no. of iterations

#flag = 0 means no critical point / weight can be found 
flag = 0 

while(k < iteration): 
    #STEP 2:compute first order partial derivate of J wrt w_0, w_1  at every w_0, w_1 values 
    j_grad_w1 = J_grad_w1(w_0,w_1) #PD wrt w_0
    j_grad_w2 = J_grad_w2(w_0,w_1) #PD wrt w_1
    
    
    #INTERMEDIATE STEP:GET UPDATED VALUE OF ALPHA 
    #coarse search for alpha using bracketing method 
    a, b = alpha_bracketing(w_0,w_1, j_grad_w1, j_grad_w2) 
    #print("in here a, b:", a, b)
    
    #fine search or optimal search for alpha
    alpha = alpha_interval_halving(a,b, j_grad_w1, j_grad_w2)    
    
    #STEP 3: compute next value of w1 from step 2
    w1_next = w_0 - alpha * j_grad_w1 
    w2_next = w_1 - alpha * j_grad_w2 
            
    #compute first order partial derivate of J at w1_next, w2_next wrt w1 and w2 
    j_grad_wnext_1 = J_grad_w1(w1_next, w2_next)  #PD wrt w1
    j_grad_wnext_2 = J_grad_w2(w1_next, w2_next)  #PD wrt w2
    
    #start computing norm of grad J vector
    j_grad_wnext_1 = j_grad_wnext_1 * j_grad_wnext_1
    j_grad_wnext_2 = j_grad_wnext_2 * j_grad_wnext_2
    
    #check if norm of grad J is within stop_value
    if math.sqrt(j_grad_wnext_1 + j_grad_wnext_2) < stop_value:
        flag = 1 #norm of grad J is within stop_value
        break    #break from loop as critical point is found
    else: #updtae w1,w2  and proceed to next iteration
        w_0 = w1_next
        w_1 = w2_next
        
        if k % 100 == 0:#these values are used for plotting on contour of J(w1, w2)
            print("Updated value of w1 from alpha search:", w_0)
            print("Updated value of w2 from alpha search:", w_1)
            print("\n")
        
        k = k+1

print("\n")        
if flag == 1:  #critical value / weights found  
    print("The bias value w_0 is:", w1_next)
    print("The slope value w_1 is:", w2_next)
    print("\nBest fit line for given dataset using Linear Regression is: ")
    print("y = " + str(w1_next) + " + " + str(w2_next) + " * x")
if flag == 0:  #critical value / weights could not be found  
    print("Start with some other initial guess")
############################## 3   



"""
Result:
Updated value of w1 from alpha search: -10.032261701106982
Updated value of w2 from alpha search: 2.97201118911932


Updated value of w1 from alpha search: -10.064925297731328
Updated value of w2 from alpha search: 2.9726191997585563




The bias value w_0 is: -10.065044200772903
The slope value w_1 is: 2.972621413057819

Best fit line for given dataset using Linear Regression is: 
y = -10.065044200772903 + 2.972621413057819 * x
"""




























"""
#PRINTING x column data 
print(x_data)#prints entire x xolumn only 
print("\n")

print(type(x_data))#prints data type as list

print(x_data[0])#prints first element of x column

print(type(x_data[0])) #prints type of ele in x column <class 'numpy.float64'>

print(len(x_data))#200

#x_data is a list of elements of type numpy.float64

for ele in x_data:
    print(ele)


#PRINTING y column data 
print(y_data)#prints entire y xolumn only 
print("\n")

print(type(y_data))#prints data type as list

print(y_data[0])#prints first element of y column

print(type(y_data[0])) #prints type of ele in y column <class 'numpy.float64'>

print(len(y_data))#200

#y_data is a list of elements of type numpy.float64


print("\n")
##being np , elements can be added using np.add(a,b)
print(np.add(x_data, y_data))
print(len(np.add(x_data, y_data)))
"""








