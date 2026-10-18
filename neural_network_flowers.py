import random
import math


# Features: 4 separate lists containing the physical dimensions of the flower (in cm)
# All values scaled between 0.0 and 1.0
sepal_length = [0.296, 0.222, 0.148, 0.111, 0.259, 0.407, 0.111, 0.259, 0.037, 0.222, 0.407, 0.185, 0.185, 0.000, 0.556,
                1.000, 0.778, 0.963, 0.444, 0.815, 0.519, 0.741, 0.222, 0.852, 0.333, 0.259, 0.593, 0.630, 0.667, 0.481]

sepal_width  = [0.750, 0.500, 0.600, 0.550, 0.800, 0.950, 0.700, 0.700, 0.450, 0.550, 0.850, 0.700, 0.500, 0.500, 1.000,
                0.600, 0.600, 0.550, 0.150, 0.400, 0.400, 0.650, 0.200, 0.450, 0.350, 0.000, 0.500, 0.100, 0.450, 0.450]

petal_length = [0.079, 0.079, 0.053, 0.105, 0.079, 0.158, 0.079, 0.105, 0.079, 0.105, 0.105, 0.132, 0.079, 0.000, 0.026,
                0.947, 0.895, 1.000, 0.763, 0.921, 0.895, 0.947, 0.579, 0.921, 0.737, 0.632, 0.816, 0.763, 0.947, 0.658]

petal_width  = [0.067, 0.067, 0.067, 0.067, 0.067, 0.200, 0.133, 0.067, 0.067, 0.000, 0.067, 0.067, 0.000, 0.000, 0.067,
                0.867, 0.933, 0.933, 0.800, 0.933, 0.800, 1.000, 0.600, 0.867, 0.933, 0.600, 0.933, 0.600, 0.867, 0.800]


# Classification Targets: 0 = Iris Setosa, 1 = Iris Versicolor
flower_type  = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


#weights and biases(4 required)

#neuron weights(1)
n1_weights=[random.uniform(-0.1,0.1),random.uniform(-0.1,0.1),random.uniform(-0.1,0.1),random.uniform(-0.1,0.1)]
n1_bias=random.uniform(-0.1,0.1)

#neuron weights(2)
n2_weights=[random.uniform(-0.1,0.1),random.uniform(-0.1,0.1),random.uniform(-0.1,0.1),random.uniform(-0.1,0.1)]
n2_bias=random.uniform(-0.1,0.1)

#neuron weights(3)
n3_weights=[random.uniform(-0.1,0.1),random.uniform(-0.1,0.1),random.uniform(-0.1,0.1),random.uniform(-0.1,0.1)]
n3_bias=random.uniform(-0.1,0.1)

#neuron weights(4)
n4_weights=[random.uniform(-0.1,0.1),random.uniform(-0.1,0.1),random.uniform(-0.1,0.1),random.uniform(-0.1,0.1)]
n4_bias=random.uniform(-0.1,0.1)


#output layer weights and biases(1)

out_weights=[random.uniform(-0.1,0.1),random.uniform(-0.1,0.1),random.uniform(-0.1,0.1),random.uniform(-0.1,0.1)]
out_bias=random.uniform(-0.1,0.1)

#learing rate
learning_rate=0.01

#sigmoid function

def sigmoid(x):
    return 1/(1 + math.exp(-x))



for epoch in range(2500):
    for sepal_len,sepal_wide,petal_len,petal_wide,target in zip(sepal_length,sepal_width,petal_length,petal_width,flower_type):

        n1_sum=(n1_weights[0]*sepal_len) + (n1_weights[1]*sepal_wide) + (n1_weights[2]*petal_len) + (n1_weights[3]*petal_wide) + n1_bias
        n1_active=sigmoid(n1_sum)

        n2_sum=(n2_weights[0]*sepal_len) + (n2_weights[1]*sepal_wide) + (n2_weights[2]*petal_len) + (n2_weights[3]*petal_wide) + n2_bias
        n2_active=sigmoid(n2_sum)

        n3_sum=(n3_weights[0]*sepal_len) + (n3_weights[1]*sepal_wide) + (n3_weights[2]*petal_len) + (n3_weights[3]*petal_wide) + n3_bias
        n3_active=sigmoid(n3_sum)

        n4_sum=(n4_weights[0]*sepal_len) + (n4_weights[1]*sepal_wide) + (n4_weights[2]*petal_len) + (n4_weights[3]*petal_wide) + n4_bias
        n4_active=sigmoid(n4_sum)

        output_sum= (n1_active*out_weights[0]) + (n2_active*out_weights[1]) + (n3_active*out_weights[2]) + (n4_active*out_weights[3]) + out_bias
        output_prediction=sigmoid(output_sum)

        error=target-output_prediction
        output_delta=error*(output_prediction*(1-output_prediction))#(a-t)a(1-a)



        n1_delta=output_delta*out_weights[0]*(n1_active*(1-n1_active)) # error for hidden layer 1 to the intial input
        n2_delta=output_delta*out_weights[1]*(n2_active*(1-n2_active))
        n3_delta=output_delta*out_weights[2]*(n3_active*(1-n3_active))
        n4_delta=output_delta*out_weights[3]*(n4_active*(1-n4_active))



        out_weights[0]+=learning_rate*output_delta*n1_active # change in output weights
        out_weights[1]+=learning_rate*output_delta*n2_active
        out_weights[2]+=learning_rate*output_delta*n3_active
        out_weights[3]+=learning_rate*output_delta*n4_active
        out_bias+=learning_rate*output_delta



        n1_weights[0]+=n1_delta*learning_rate*sepal_len
        n1_weights[1]+=n1_delta*learning_rate*sepal_wide
        n1_weights[2]+=n1_delta*learning_rate*petal_len
        n1_weights[3]+=n1_delta*learning_rate*petal_wide
        n1_bias+=learning_rate*n1_delta


        n2_weights[0]+=n2_delta*learning_rate*sepal_len
        n2_weights[1]+=n2_delta*learning_rate*sepal_wide
        n2_weights[2]+=n2_delta*learning_rate*petal_len
        n2_weights[3]+=n2_delta*learning_rate*petal_wide
        n2_bias+=learning_rate*n2_delta


        n3_weights[0]+=n3_delta*learning_rate*sepal_len
        n3_weights[1]+=n3_delta*learning_rate*sepal_wide
        n3_weights[2]+=n3_delta*learning_rate*petal_len
        n3_weights[3]+=n3_delta*learning_rate*petal_wide
        n3_bias+=learning_rate*n3_delta

        n4_weights[0]+=n4_delta*learning_rate*sepal_len
        n4_weights[1]+=n4_delta*learning_rate*sepal_wide
        n4_weights[2]+=n4_delta*learning_rate*petal_len
        n4_weights[3]+=n4_delta*learning_rate*petal_wide
        n4_bias+=learning_rate*n4_delta


    if epoch%200 ==0:
         print("Target :",target,"Prediction :",output_prediction)       




print("\n--- Model Training Complete! ---")
