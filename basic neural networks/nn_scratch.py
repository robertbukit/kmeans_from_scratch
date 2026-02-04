# Apa yang dilakukan disini: Membuat Model Neural Network Sederhana (FeedForward Model)
# Bagaimana struuktur modelnya: 1. Input Layer (banyknya neuron tergantung jumlah variabel)
# 2. Hidden Layer (terdiri dari beberap neuron), 3. Output
# no library, no numpy, no pytorch, no tensorflow, dsb
# import packages yang diperlukan untuk membangun keseluruhan model neural networks

import math
import random

# Step 1: Definisikan fungsi aktivasinya dulu

# disini saya pakai fungsi sigmoid
# fungsi sigmoid memetakan nilai ke rentang 0 - 1
# sigma(x) = 1 / (1 + exp(-z))
# Tapi umumnya fungsi ReLU sangat banyak digunakan karena menghindari vanishing gradient

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# kemudian definisikan dahulu juga untuk turunan fungsi sigmoidnya
# karena pada backpropagation kita memerllukan gradient

def sigmoid_derivative(x):
    sx = sigmoid(x)
    return sx * (1 - sx)        # Perhitungan terlampir di pdf

'''
# Step 2: Membuat Struktur Neural Networks

# 2 input neurons, 1 hidden layer dengan 2 neuoron, 1 output (0/1)
# yang kita perlukan:
# Bobot W1 (matriks) sebagai parameter trainable dari input layer, dan Bias B1 (vektor)
# Bobot W2 (matriks) sebagai parameter trainable dari input layer, dan Bias B2 (vektor)
'''

# inisiasi bobot dengan nilai acak yang kecil
# bobot di set random dengan rentang nilai -1 sampai 1

W_h = [
    [random.uniform(-1, 1), random.uniform(-1, 1)],     
    [random.uniform(-1, 1), random.uniform(-1, 1)]
]

B_h = [random.uniform(-1, 1), random.uniform(-1, 1)]

W_o = [random.uniform(-1, 1), random.uniform(-1, 1)]
B_o = random.uniform(-1, 1)


# Implementasi matriks bobot tsb bergantung pada jumlah variabel/fitur pada datanya
# Namun untuk setiap variabel/fitur nya bobot di hitung sama untuk semua entry datanya
# Misal dataset terdiri dari 1000 baris data dan 10 kolom, maka matriks bobotnya terdiri dari 
# 10 baris dan beberapa kolom (untuk neuron dalam hidden layernya) dan vektor bias tergantung dari banyaknya neuron dalam hidden layer

''''
# Step 3: Forward pass

# Tujuan yg di set pada algoritma forward ini adalah mengirimkan input melalui layer network
# menghitung bobot untuk semua data dan menerapkan fungsi aktivasi di hidden layer dan output
# z1 adalah penjumlahan bobot input untuk neuron 1 di hidden layer
# => input 1 * bobot 1 + input 2 * bobot 2 + input n * bobot n + b (dihitung kembali ke semua neuron utk mendapat bobot yg berbeda)
#    tiap input berisi misalnya ribuan entry data dan labelnya, semua nya dihitung berulang sama seperti diatas sampai mendapat output dan evaluasi dgn loss func
'''

def forward(x):
    # Hidden layer
    z1 = x[0] * W_h[0][0] + x[1] * W_h[1][0] + B_h[0]
    a1 = sigmoid(z1)

    z2 = x[0] * W_h[0][1] + x[1] * W_h[1][1] + B_h[1]
    a2 = sigmoid(z2)

    # Output Layer
    z3 = a1 * W_o[0] + a2 * W_o[1] + B_o
    a3 = sigmoid(z3)

# forward pass ini membuat proses untuk mendapat hasil prediksi, tapi hasilnya masih permulaan
# kita memerlukan hasil prediksi awal untuk dapat kita latih dari fungsi kerugian pertama


'''
# Step 4: Loss Function

# kita menghitung kuadrat residu untuk mendapatkan nilai fungsi kerugian (MSE)
# catatan: output akan berbohong di rentang 0 dan 1, yg nilai seharusnya 0 dan 1
# nilai residu akan bernilai negatif jika label sebenarnya lebih kecil dari prediksi
# oleh karena itu MSE mengkuadratkan nilai residunya agar error tetap positif
'''

def loss_func(y_true, y_pred):
    return (y_true - y_pred)**2

'''
# Step 5: Backpropagation

# sekarang kita harus menyesuaikan bobotnya, maka error akan menjadi sangat kecil
# umumnya kita harus mengiterasikannya cukup banyak, dengan itu loss menjadi konvergen ke 0
# untuk itu pada algoritma backpropagation ini kita menghitung derivasi loss function nya terhadap bobot dan bias
# algoritma ini disebut gradient, jadi kita memulai kembali perhitungan berlawanan dari akhir ke awal
'''

def backward(x, y_true, learning_rate):
    global W1, B1, W2, B2

    z1, a1, z2, a2, z3, a3 = forward(x)  

    # hitung semua derivasinya
    dL_da3 = -2 * (y_true - a3)
    da3_dz3 = sigmoid_derivative(z3)

    dL_dW2 = [
        dL_da3 * da3_dz3 * a1,
        dL_da3 * da3_dz3 * a2
    ]
    dL_dB2 = dL_da3 * da3_dz3

    dL_dz1 = dL_da3 * da3_dz3 * W2[0] * sigmoid_derivative(z1)
    dL_dz2 = dL_da3 * da3_dz3 * W2[1] * sigmoid_derivative(z2)

    dL_dW1 = [
        [dL_dz1 * x[0], dL_dz2 * x[0]],
        [dL_dz1 * x[1], dL_dz2 * x[1]]
    ]
    dL_dB1 = [dL_dz1, dL_dz2]

    for i in range(2):
        for j in range(2):
            W1[i][j] -= learning_rate * dL_dW1[i][j]
    for i in range(2):
        B1[i] -= learning_rate * dL_dB1[i]
    for i in range(2):
        W2[i] -= learning_rate * dL_dW2[i]
    B2 -= learning_rate * dL_dB2
    

'''
# Step 6: Train data dan parameter

# kita ingin tahu bagaimana model ini bekerja setiap iterasinya
# disini kita mendefisikan dulu seberapa banyak iterasinya dan seberapa jauh langkah belajarnya (derivasi) atau learning rate nya
# learning tidak boleh terlalu besar (karena tidak akan konvergen nantinya)
# learning rate juga tidak boleh terlalu kecil (terlalu mahal secara komputasi dan resiko overfit)
# disini kita pakai learning rate umum yaitu 0,1 dan epoch nya 10000
'''

learning_rate = 0.1
epochs = 10000

# sekarang kita mendefinisikan train data dengan dummy data

# training data
data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0)
]

for epoch in range(epochs):
    for x, y in data:
        backward(x, y, learning_rate)

    # menunjukkan setiap 1000 epochs
    if epoch %  1000 == 0:
        total_loss = 0
        for x, y in data:
            y_pred = forward(x)[-1]  
            total_loss += (y - y_pred) ** 2
        print(f"Epoch {epoch} | Loss: {total_loss:.4f}")



# step 8: testting 

runs = 1000  # berapa kali input di test

print("\nTest after training:")
for x, y in data:
    correct = 0
    for _ in range(runs):
        y_pred = forward(x)[-1]  
        y_class = 1 if y_pred >= 0.5 else 0  
        if y_class == y:
            correct += 1
    print(f"Input: {x} → Expected: {y} | Correct: {correct}/{runs} ({correct/runs*100:.1f}%)")
    print(f"Input: {x}, y_pred: {y_pred:.10f}, rounded: {y_class}, expected: {y}")