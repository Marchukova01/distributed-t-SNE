# distributed-t-SNE
Реализация алгоритма t-SNE, позволяющая избежать хранения объема информации, квадратично зависящего от объема входных данных.  
1. Эксперименты
Рассмотрим вложения, получающееся в результате работы распределенного
алгоритма, а также t-SNE, реализованного в библиотеке scikit-learn. Тестирование проводилось на примере наборов точек, с заранее известным разделением на
кластеры
<img width="374" alt="image" src="https://github.com/Marchukova01/distributed-t-SNE/assets/90204625/8b4520b2-a428-47e4-8677-a15234fffcb2">  
<img width="780" alt="image" src="https://github.com/Marchukova01/distributed-t-SNE/assets/90204625/e177967e-cdf0-4985-ba94-fa930a31694a"> 
<img width="375" alt="image" src="https://github.com/Marchukova01/distributed-t-SNE/assets/90204625/8a359be0-245b-411a-8d3b-e8b133f1efe3">  
<img width="774" alt="image" src="https://github.com/Marchukova01/distributed-t-SNE/assets/90204625/314b39a5-fd04-4e3e-b07e-6de2f08eb31d">  
<img width="376" alt="image" src="https://github.com/Marchukova01/distributed-t-SNE/assets/90204625/de60af72-c0f0-45f3-b29b-285372aa399c"> 
<img width="787" alt="image" src="https://github.com/Marchukova01/distributed-t-SNE/assets/90204625/eef150dd-b399-4010-82eb-f3f258fa5c39">  
  
2. MNIST  
База данных MNIST состоит из нормализованных по размеру и расположенных
в центре изображения фиксированного размера черно-белых образцов рукописных
цифр. Датасет содержит 60000 изображений для обучения и тестовый набор из
10000 экземпляров. Каждое изображение имеет размер 28×28 = 784 пикселя. 
Примеры изображений, содержащихся в датасете MNIST приведены на Рис. 7.    
<img width="453" alt="image" src="https://github.com/Marchukova01/distributed-t-SNE/assets/90204625/899bc5dc-f4df-4bba-ba93-77c26f9988fc">  
<img width="382" alt="image" src="https://github.com/Marchukova01/distributed-t-SNE/assets/90204625/09fb2690-8b22-43a3-8dd7-8de4fd70e985">



