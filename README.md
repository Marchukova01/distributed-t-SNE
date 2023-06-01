# distributed-t-SNE
Реализация алгоритма t-SNE, позволяющая избежать хранения объема информации, квадратично зависящего от объема входных данных.
Рассмотрим вложения, получающееся в результате работы распределенного
алгоритма, а также t-SNE, реализованного в библиотеке scikit-learn. Тестирование проводилось на примере наборов точек, с заранее известным разделением на
кластеры
<img width="374" alt="image" src="https://github.com/Marchukova01/distributed-t-SNE/assets/90204625/8b4520b2-a428-47e4-8677-a15234fffcb2">  
<img width="378" alt="image" src="https://github.com/Marchukova01/distributed-t-SNE/assets/90204625/da4b07b4-4683-4c33-9128-90ad5fb5f929">  
<img width="375" alt="image" src="https://github.com/Marchukova01/distributed-t-SNE/assets/90204625/8a359be0-245b-411a-8d3b-e8b133f1efe3">  
<img width="375" alt="image" src="https://github.com/Marchukova01/distributed-t-SNE/assets/90204625/832df3a0-aa9a-47b0-8b98-4541b1fb05b8">  
<img width="376" alt="image" src="https://github.com/Marchukova01/distributed-t-SNE/assets/90204625/de60af72-c0f0-45f3-b29b-285372aa399c">  
<img width="452" alt="image" src="https://github.com/Marchukova01/distributed-t-SNE/assets/90204625/d4401e70-d26e-45e3-9c00-f825e3a7bc9c">  
База данных MNIST состоит из нормализованных по размеру и расположенных
в центре изображения фиксированного размера черно-белых образцов рукописных
цифр. Датасет содержит 60000 изображений для обучения и тестовый набор из
10000 экземпляров. Каждое изображение имеет размер 28×28 = 784 пикселя. 
Примеры изображений, содержащихся в датасете MNIST приведены на Рис. 7.  
<img width="453" alt="image" src="https://github.com/Marchukova01/distributed-t-SNE/assets/90204625/899bc5dc-f4df-4bba-ba93-77c26f9988fc">  


