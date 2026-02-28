[Türkçe] Felçli ve ALS Hastalarının İletişimi İçin Geliştirilmiştir 
Merhaba
Bu projede, derin öğrenme ve bilgisayarlı görü algoritmaları kullanarak göz kırpma hareketlerini gerçek zamanlı olarak algılayan ve Mors alfabesi üzerinden metne dönüştüren uçtan uca bir sistem geliştirdim.
Proje Detayları:
🔹 Model Mimarisi: Yüz ve göz tespiti için OpenCV Haar Cascade yapısını ve gözün açık/kapalı durumunu sınıflandırmak için Evrişimli Sinir Ağları (CNN) katmanlarını birleştiren bir yapı tasarladım.
🔹 Amaç: ALS ve felçli hastalar gibi motor beceri kaybı yaşayan bireylerin iletişimini, göz hareketlerini otonom bir şekilde metne çevirerek kolaylaştırmak.
🔹 Teknik Süreç ve Dağıtım:Görüntü İşleme: Kırpılan göz görüntüleri 64x64 boyutunda gri tonlamalı matrislere dönüştürülerek $1/255$ oranında normalize edildi.
Katman Yapısı: Göz durumunu saptamak için Conv2D ve MaxPooling2D katmanları, nihai sınıflandırma için Flatten ve Dense katmanları kullanıldı.
Gerçek Zamanlı Çeviri: Algılanan kısa ve uzun göz kırpma süreleri analiz edilerek otonom bir şekilde anlık Mors kodu ve metin üretimi sağlandı.
🔹 Performans: Model, kamera akışı üzerinden yüksek doğrulukla anlık tepki verecek şekilde optimize edildi.
Kullanılan Teknolojiler: Python, Keras, TensorFlow, OpenCV.
Kodları incelemek ve geliştirme önerilerinizi paylaşmak isterseniz geri bildirimleriniz benim için çok değerli!
Yazar: Mustafa Alpergün

[English] Developed Specifically for the Communication of Paralyzed and ALS Patients 
HelloIn 
this project, I developed an end-to-end real-time system that detects eye blink patterns using computer vision and deep learning algorithms, translating them into text via Morse code.
Project Overview:
🔹 Model Architecture: I designed a model combining OpenCV Haar Cascades for robust face/eye detection and Convolutional Neural Networks (CNN) for high-accuracy eye state classification.
🔹 Objective: To facilitate communication for individuals with motor skill impairments, such as ALS and paralyzed patients, by autonomously converting eye movements into text.
🔹 Technical Pipeline:
Image Engineering: Processed cropped eye images into 64x64 grayscale arrays with $1/255$ normalization.
Layer Composition: Integrated Conv2D and MaxPooling2D layers followed by Flatten and Dense layers to classify open and closed eye states.
Real-Time Translation: Extracted chronometric data from blink durations (short/long) to autonomously generate and display Morse code and text instantly.
🔹 Evaluation: The model was optimized to perform highly accurate real-time analysis over live video feeds.
Tech Stack: Python, Keras, TensorFlow, OpenCV.
Feel free to review the code and share your feedback!
Author: Mustafa Alpergün
