#!/usr/bin/env python
# coding: utf-8

# # Web Editörleri Ara Sınav Uygulaması
# 
# ### Talimatlar:
# * <font color=red>Bilgisayarın Görmesi (Computer Vision)</font> ile ilgili OpenCV kütüphanesinin kullanımını anlatan bir uygulama kılavuzu hazırlayınız.
# * Klavuzda yer alan başlık ve içerik yapısı aşağıdaki gibi olacaktır.
# * Dosyayı hazırladıktan sonra Github üzerinde bir hesap açıp, bu dosyayı oraya yükleyip, web adresini bu hücrece verilen yere yazınız.
# * Bu dosyayı ayrıca DBS üzerinde açılan sınav ödevi etkinliğine yükleyiniz.
# * Bu dosyayı hazırladıktan sonra DBS'ye yüklerken mutlaka dosya adını değiştirerek Kendi Adınızı yazınız. Github'a yüklerken dosya adını konu başlığı ile değiştiriniz.
# 
# 
# 

# # Computer Vision
# 
# Computer vision, bilgisayar sistemlerinin görsel bilgiyi algılama ve işleme yeteneğidir. Bu alan, bilgisayar biliminin yanı sıra yapay zeka, makine öğrenmesi, ve görüntü işleme tekniklerini içerir. Computer vision, çeşitli sensörler ve kameralar aracılığıyla elde edilen görsel verileri analiz ederek, bu verilerden anlam çıkarmayı amaçlar. Bu süreç, insan gözünün gördüğü bir görüntüyü beyinle nasıl işlediğine benzer mekanizmalarla çalışır. Görüntülerden elde edilen veriler, çeşitli algoritmalar ve modeller kullanılarak işlenir, bu sayede makineler çevrelerini anlayabilir ve belirli görevleri otomatik olarak yerine getirebilir.
# 
# Bu teknolojinin uygulama alanları oldukça geniştir. Örneğin, sanayide otomatik montaj hatlarında parçaların doğru şekilde yerleştirilip yerleştirilmediğini kontrol etmek için kullanılır. Tıp alanında, radyoloji görüntülerini analiz ederek hastalıkların teşhisine yardımcı olur. Güvenlik sistemlerinde, izleme kameralarından gelen verileri analiz ederek şüpheli faaliyetleri tespit edebilir. Otomotiv sektöründe ise, sürücüsüz araçların çevresini algılayarak trafikte güvenli bir şekilde hareket etmelerini sağlar. Bu örnekler, computer vision'ın nasıl geniş bir yelpazede kullanılabileceğini göstermektedir.
# 
# Computer vision algoritması geliştirme süreci, büyük miktarda etiketlenmiş görüntü verisinin toplanmasıyla başlar. Bu veriler, öğrenme modellerinin eğitilmesi için kullanılır. Bu süreçte en çok başvurulan yöntemlerden biri, evrişimli sinir ağları (CNN) teknolojisidir. CNN'ler, görüntü içindeki desenleri, kenarları ve diğer özellikleri otomatik olarak tanıyabilen ve bu bilgileri sınıflandırma veya tanıma gibi daha karmaşık işlemler için kullanabilen derin öğrenme modelleridir. Bu modeller, makinelerin insan gözüyle görebileceği detayları yakalamasını ve hatta bazen insanlardan daha iyi performans göstermesini sağlayabilir.
# 
# Bu teknolojinin en büyük zorluklarından biri, algoritmaların karmaşık ve dinamik gerçek dünya ortamlarında doğru ve etkili bir şekilde çalışabilmesini sağlamaktır. Örneğin, farklı ışık koşulları, görüş açıları veya kapalı alanlar gibi değişken ortamlarda, algoritmaların güvenilirliğini koruması zor olabilir. Bunun yanı sıra, gizlilik ve etik konuları da büyük önem taşır; özellikle kişisel verilerin işlenmesi ve yüz tanıma teknolojileri gibi uygulamalar, ciddi gizlilik ve etik sorunlarını beraberinde getirebilir.
# 
# Sonuç olarak, computer vision, hem akademik araştırmalar hem de endüstriyel uygulamalar açısından önemli bir alan olmaya devam etmektedir. Sürekli gelişen teknolojiler ve algoritmalar sayesinde, bu alanın kapasitesi ve kullanım alanları da genişlemektedir. Yapay zeka ve otomasyonun giderek daha fazla entegre olduğu bir dünyada, computer vision'ın önemi ve etkisi artarak devam edecektir, ve bu da yeni teknolojik ilerlemelerle insan hayatını daha da kolaylaştırabilir.

# ### OpenCv Kütüphanesi Kullanımı
# OpenCv kütüphanesi kullanarak aşağıdaki işlemler yapılabilir:
# 
# OpenCV (Open Source Computer Vision Library) kütüphanesi, gerçek zamanlı bilgisayar görüsü uygulamalarını desteklemek amacıyla geliştirilmiş açık kaynaklı bir kütüphanedir. Python, C++, Java ve diğer dillerde kullanılabilir. Çeşitli görüntü işleme ve bilgisayar görüsü tekniklerinin uygulanmasını kolaylaştırır. İşte OpenCV kullanarak gerçekleştirebileceğiniz üç temel işlem:
# 
# 1. **Görüntü İşleme**: OpenCV, görüntü işleme için geniş bir fonksiyon seti sunar. Bu işlevler arasında renk dönüşümleri, kenar bulma, bulanıklaştırma ve görüntü filtreleme yer alır. Örneğin, bir görüntüyü gri tonlamaya çevirmek, görüntü üzerinde Gaussian Blur uygulamak veya Canny Edge Detection kullanarak bir görüntünün kenarlarını tespit etmek mümkündür. Bu işlemler, görüntü üzerindeki istenmeyen gürültüyü azaltmaya ve önemli özelliklerin daha iyi tanınmasına yardımcı olur.
# 
# 2. **Nesne Tespiti ve Tanıma**: OpenCV, Haar Cascade Classifier, HOG + Linear SVM ve son zamanlarda popüler olan derin öğrenme tabanlı yöntemler dahil olmak üzere çeşitli nesne tespit algoritmalarını destekler. Bu algoritmalar kullanılarak yüzler, insanlar, araçlar gibi nesneler gerçek zamanlı olarak tespit edilebilir ve tanımlanabilir. Özellikle, yüz tanıma ve otomatik plaka tanıma sistemlerinde sıkça kullanılır.
# 
# 3. **Video İşleme ve Hareket Analizi**: OpenCV, video akışları üzerinde işlem yapma yeteneği sunar. Bu, video dosyalarını veya kamera akışlarını okuma, bu akışlar üzerinde işlem yapma ve sonuçları kaydetme yeteneği içerir. Hareket tespiti, nesne takibi (tracking) ve optik akış gibi teknikler, video içindeki nesnelerin hareketlerini analiz etmek için kullanılabilir. Bunun dışında, arka plan çıkarma ve arka planın gerçek zamanlı olarak güncellenmesi gibi işlemler de mümkündür.
# 
# 4. **Yüz Tespiti**: OpenCV, yüz tespiti için önceden eğitilmiş Haar cascade modelleri sunar. Bu modeller, görüntülerdeki veya video akışlarındaki insan yüzlerini hızlı ve etkili bir şekilde tespit etmek için kullanılabilir. Yüz tespiti, güvenlik sistemlerinden sosyal medya uygulamalarına kadar birçok alanda kullanılır. Örneğin, bir güvenlik kamerası sistemine entegre edilerek belirli alanlara girmesi istenmeyen kişileri tespit etmek için kullanılabilir.
# 
# 5. **Gerçek Zamanlı Nesne Takibi**: Nesne takibi, video içinde hareket eden bir nesnenin konumunu anlık olarak izlemek için kullanılır. OpenCV, KCF (Kernelized Correlation Filters) gibi algoritmaları kullanarak nesne takibi yapabilir. Bu özellik, otomasyon sistemlerinde, spor analizlerinde veya interaktif projelerde nesnelerin hareketlerini izlemek için kullanılabilir.
# 
# 6. **Trafik İşaretleri Tanıma**: Yol işaretlerini tanıma, sürücüler için önemli bir güvenlik özelliğidir. OpenCV ile trafik işaretlerinin tespit edilmesi ve tanınması, görüntü işleme ve makine öğrenmesi teknikleri kullanılarak gerçekleştirilebilir. Bu, özellikle otonom araçlar için kritik bir uygulamadır.
# 
# 7. **Arkaplan Çıkarma**: OpenCV, bir videodaki hareketli nesneleri sabit bir arka plandan ayırmak için kullanılabilir. Bu işlem, güvenlik kamerası görüntülerinde istenmeyen hareketlerin tespiti, spor videolarında oyuncuların analizi veya video düzenlemede özel efektler eklemek için kullanılabilir.
# 
# 8. **Optik Karakter Tanıma (OCR)**: OpenCV, görüntülerdeki metinleri tanıma ve okuma işlemleri için de kullanılabilir. OCR uygulamaları, doküman tarama, plaka tanıma veya reklam panolarından bilgi çekme gibi çeşitli amaçlar için kullanılabilir.
# 
# 9. **3D Nesne Rekonstrüksiyonu**: Stereoskopik kameralar kullanarak, OpenCV ile çevredeki nesnelerin 3D modelleri oluşturulabilir. Bu, sanal gerçeklik uygulamaları, robotikte çevresel algılama veya tıbbi görüntüleme gibi alanlarda kullanılabilir.
# 
# Bu örnekler, OpenCV'nin geniş kullanım alanlarını göstermektedir ve bu teknoloji, sürekli gelişen bilgisayar görüşü alanında çığır açıcı yenilikler sunmaktadır.

# ### 0- Kütüphanenin kurulumu

# In[ ]:


get_ipython().system('pip install --upgrade pip')

get_ipython().system('conda update --all -y')

get_ipython().system('pip install opencv-contrib-python')

get_ipython().system('pip install opencv-python-headless matplotlib')

import cv2
print(cv2.__version__)


# # 1-# Adım 1: pip'i Güncelleme
# Python paket yöneticisi olan pip, sık sık güncellenir. Güncel bir pip sürümü kullanmak, bağımlılıkların daha düzgün yönetilmesini ve hata ihtimallerinin azalmasını sağlar. Jupyter Notebook'ta pip'i güncellemek için aşağıdaki komutu kullanabilirsiniz:

# !pip install --upgrade pip
# 

# ### 2- #Adım 2: OpenCV'nin Ana Paketini Yüklemek
# Opencv-python, OpenCV kütüphanesinin temel fonksiyonlarını içeren pakettir. Bu paket, çoğu temel görüntü işleme ve bilgisayar görüşü işlemleri için gereklidir. Yükleme için aşağıdaki komut kullanılır:

# In[ ]:


get_ipython().system('pip install opencv-python')
get_ipython().system('pip install opencv-python-headless')
get_ipython().system('pip install opencv-contrib-python-headless')




# 3- #Adım: Ekstra Modüllerle OpenCV'yi Yüklemek
# Opencv-contrib-python paketi, OpenCV'nin bazı ekstra modüllerini içerir. Bu ek modüller, genişletilmiş özelliklere erişim sağlar ancak her projede gerekli olmayabilir. Bu paketi yüklemek için aşağıdaki komutu kullanın:

# In[ ]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# #Adım 4: OpenCV Sürümünü Kontrol Etme
# Kurulumun başarılı olduğundan emin olmak ve yüklenen OpenCV sürümünü görmek için Python kodu kullanarak sürümü yazdırabilirsiniz. Aşağıdaki kod parçasını çalıştırarak yüklenen sürümü kontrol edebilirsiniz:

# In[ ]:


import cv2
print(cv2.__version__)


# 5** Yüz tanıma işlevleri tanıtılır**
# 

# In[ ]:


# Kamera akışını başlatal
def start_camera(index=0):
    """ Kamera akışını başlatma """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise IOError("Kamera açılamıyor!")
    return cap

# Görüntü üzerinde yüz tanıma
def detect_faces(frame, face_cascade):
    """ Görüntü üzerinde yüz tanıma """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return frame

# Kamera görüntüsünü gösterme

def display_image(frame):
  # Görüntüyü RGB formatına çevir (OpenCV varsayılan olarak BGR formatında okur)
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_image)
    plt.title("Kamera Görüntüsü")
    plt.axis('off')
    plt.show()


# 6** Yüz Tanıma Modelini Yükleme**
# 

# In[ ]:


# Haarcascade XML dosyasını yükle
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# 7** Kamera Akışından Yüzleri Tanıma ve Gösterme**
# 

# In[ ]:


# Kamera başlat
cap = start_camera()

try:
    while True:
        # Kamera'dan bir frame oku
        ret, frame = cap.read()
        
        # Frame üzerinde yüz tespiti yap
        if ret:
            frame_with_faces = detect_faces(frame, face_cascade)
            display_image(frame_with_faces)
        else:
            print("Kamera'dan görüntü alınamıyor!")
        input("Devam etmek için Enter'a basın. Çıkmak için ise Ctrl+C yapın.")
finally:
    # Kaynakları serbest bırak
    cap.release()
    cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




