# coding=utf-8
import numpy, scipy.special, matplotlib.pyplot, csv, math
from time import time

# Создано после прочтения книги Тарика Рашида "Создаём нейронную сеть"
def load_file(file_path):
    
    # Открываем файл с данными и копируем всё в data_list
    data_file = open(file_path, 'r')
    # Считываем весь файл в переменную
    data_list = data_file.readlines()
    # Закрытваем файл
    data_file.close()
    return data_list


def neuralNet_research(hidden_nodes = {100:1, 200:2, 300:3},learning_rate = {0.1:1, 0.125:2, 0.15:3, 0.175:4, 0.2:5},eras = 30):
#--------------------------ОБУЧЕНИЕ--------------------------------------------
    # Загружаем обучающий массив в data_list
    data_list = load_file('mnist/mnist_train_100.csv')
    data_list_test = load_file('mnist/mnist_test.csv')
    output_char = numpy.zeros((4,450))
    tic_train=time()
    global_iterator = 0
    for i in hidden_nodes:
        for j in learning_rate:
            Network=neuralNet(Input_nodes,i,Output_nodes,j)
            for era in range(eras):
                for line in data_list:
                    # Преобразуем i-ую строку в массив данных используя запятую, как разделитель
                    all_values = line.split(',')
                    # Нормируем массив и выкидываем идентификатор строки(первый символ)
                    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                    # Задаём пустой целевой массив для нейросети
                    targets = numpy.zeros(Output_nodes) + 0.01
                    # Задаем "выходной нейрон"
                    targets[int(all_values[0])] = 0.99
                    # Запускаем обучение 
                    Network.train(inputs, targets)
    
#--------------------------ТЕСТИРОВАНИЕ----------------------------------------
                
                precision = Network.test(data_list_test)
                output_char[0][global_iterator] = i
                output_char[1][global_iterator] = j
                output_char[2][global_iterator] = era
                output_char[3][global_iterator] = precision
                global_iterator+=1
                print(output_char)
                
    toc_train=time()
    
    print("Время тренировки: " + str((round((toc_train - tic_train),1))/60.0) + 'мин.\n')
    max_efficiency=numpy.argmax(output_char[3])
    print("Максимальная эффективность равна: "+ str((output_char[3][max_efficiency])*100)+'%.')
    print('Параметры сети при максимальной эффективности:')
    print('Количество скрытых слоёв: '+str(output_char[0][max_efficiency]) )
    print('Коэффициент обучения: '+str(output_char[1][max_efficiency]) )
    print('Количество эпох: '+str(output_char[2][max_efficiency]) )
    
    
#-------------------------КЛАСС НЕЙРОННОЙ СЕТИ---------------------------------
class neuralNet:
    
    # Инициализация нейронной сети
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # задаём количество узлов во входном, скрытом и выходном слоях
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # коэффициент обучения
        self.lr = learningrate
        
        # wih (weight input-hidden) - матрица весовых коэффициентов между
        # входным и скрытым слоями
        # who (weight hidden-output) - между скрытым и выходным слоями
        # инциализация весов происходит по нормальному закону 
        self.wih = numpy.random.normal(0.0, pow(self.hnodes,-0.5),(self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes,-0.5),(self.onodes, self.hnodes))
        
        # Функция активации
        self.activation_function = lambda x: scipy.special.expit(x)
        
    
    
    # Тренировка сети
    def train(self, inputs_list, targets_list):
         # преобразуем входной массив данных в двумерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets =numpy.array(targets_list, ndmin=2).T
        
        # Расчёт входящих сигналов для скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs) # dot - перемножение матриц
        # Расчёт исходящих сигналов для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # Расчёт входящих сигналов для выходного слоя
        final_inputs = numpy.dot(self.who, hidden_outputs) # dot - перемножение матриц
        # Расчёт исходящих сигналов для выходного слоя
        final_outputs = self.activation_function(final_inputs)
        
        # Выходная ошибка = целевое значение - 
        output_errors = targets - final_outputs
        
        # Ошибки скрытого слоя - это выходные ошибки распределённые 
        # пропорционально весам и рекомбинированные на скрытых узлах
        hidden_errors = numpy.dot(self.who.T, output_errors)
        
        # Обновление весовых коэффициентов связей между скрытым и выходным слоями
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        # Обновление весовых коэффициентов связей между скрытым и выходным слоями
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        
        
        
    # Опрос сети
    def query(self, inputs_list):
        # Преобразуем входной массив данных в двумерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # Расчёт входящих сигналов для скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs) # dot - перемножение матриц
        # Расчёт исходящих сигналов для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # Расчёт входящих сигналов для выходного слоя
        final_inputs = numpy.dot(self.who, hidden_outputs) # dot - перемножение матриц
        # Расчёт исходящих сигналов для выходного слоя
        final_outputs = self.activation_function(final_inputs)
        
        #print('Входные данные:\n'+str(inputs))
        #print('Выходные данные:\n'+str(final_outputs))

        return final_outputs
    
            
    # Сохранение весов в CSV-файл по указанному пути
    def save_weight(self,file_path_wih='wih.csv', file_path_who='who.csv'):
        
        numpy.savetxt(file_path_wih, self.wih, delimiter=",")
        numpy.savetxt(file_path_who, self.who, delimiter=",")
        
        print('Весовые коэффициенты сохранены.')
        
        
    # Загрузка весов сети
    def load_weight(self, file_path_wih='wih.csv', file_path_who='who.csv'):
        # Подгружаем весовые коэффициенты input-hidden по указанному пути
        self.wih = numpy.genfromtxt(file_path_wih,delimiter=',')
        # # Вычисляем количество столбцов матрицы и задаём входные узлы
        self.inodes = (self.wih.shape)[1]
        # Вычисляем количество строк матрицы и задаём скрытые узлы
        self.hnodes = (self.wih.shape)[0]
        
        # Подгружаем весовые коэффициенты hidden-output по указанному пути
        self.who = numpy.genfromtxt(file_path_who,delimiter=',')
        # # Вычисляем количество столбцов матрицы и задаём входные узлы
        self.inodes = (self.who.shape)[0]
        
        print('Весовые коэффициенты загружены.')
        
    
    def query_reverse(self, inputs_list):
        # Задаём функцию обратную сигмоде
        reverse_activation_function = lambda x: scipy.special.logit(x)
        # Преобразуем входной массив данных в двумерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # Расчёт входящих сигналов для скрытого слоя
        hidden_inputs = numpy.dot(self.who.T, inputs) # dot - перемножение матриц
        hidden_inputs -= numpy.min(hidden_inputs)
        hidden_inputs /= numpy.max(hidden_inputs)
        hidden_inputs *= 0.98
        hidden_inputs += 0.01
        # Расчёт исходящих сигналов для скрытого слоя
        hidden_outputs = reverse_activation_function(hidden_inputs)
        # Расчёт входящих сигналов для выходного слоя
        final_inputs = numpy.dot(self.wih.T, hidden_outputs) # dot - перемножение матриц
        final_inputs -= numpy.min(final_inputs)
        final_inputs /= numpy.max(final_inputs)
        final_inputs *= 0.98
        final_inputs += 0.01
        # Расчёт исходящих сигналов для выходного слоя
        final_outputs = reverse_activation_function(final_inputs)
        
        image_array = numpy.asfarray( final_outputs[:]*255).reshape(28,28)
        
        return image_array
    
    def test(self, data_list_test):
        # Журнал оценок работы сети
        scorecard = []
        for line in data_list_test:
            # Преобразуем i-ую строку в массив данных используя запятую, как разделитель
            all_values = line.split(',')
            # Правильный ответ - первое значение
            correct_label = int(all_values[0])
            # Нормируем массив и выкидываем идентификатор строки(первый символ)
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # Опрашиваем сеть и записываем её ответ в outputs
            outputs = self.query(inputs)
            # Индекс наибольшего значения - маркерное значение
            label = numpy.argmax(outputs)
            # Если маркер совпадает с правильным ответом - добавляем в scorecard 1, если нет - 0
            if (label == correct_label):
                scorecard.append(1)
            else:
                scorecard.append(0)
        return ((numpy.asarray(scorecard)).sum())/10000.0


#----------------------ЗАДАНИЕ ПАРАМЕТРОВ--------------------------------------

Input_nodes = 784
Hidden_nodes = 300  #300 --> max
Output_nodes = 10
eras = 10      # 10 --> max

# Коэффициент обучения
Learning_rate =0.1  #--> max 0.1

data_list = load_file('mnist/mnist_train.csv')
data_list_test = load_file('mnist/mnist_test.csv')
 
Network = neuralNet(Input_nodes,Hidden_nodes,Output_nodes,Learning_rate)

Network.load_weight()
#for era in range(eras):
#    for line in data_list:
#        # Преобразуем i-ую строку в массив данных используя запятую, как разделитель
#        all_values = line.split(',')
#        # Нормируем массив и выкидываем идентификатор строки(первый символ)
#        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
#        # Задаём пустой целевой массив для нейросети
#        targets = numpy.zeros(Output_nodes) + 0.01
#        # Задаем "выходной нейрон"
#        targets[int(all_values[0])] = 0.99
#        # Запускаем обучение 
#        Network.train(inputs, targets)


precision = Network.test(data_list_test)
print("Точность сети: "+str(precision*100)+'%')

#Network.save_weight()
#-----------------------СОХРАНЕНИЕ СЕТИ----------------------------------------

#with open("wih.csv", "w", newline='') as file:
#    csv.writer(file).writerow(Network.wih)
#    
#with open("who.csv", "w", newline='') as file:
#    csv.writer(file).writerow(Network.who)
#    
    
    
    
    
    
    