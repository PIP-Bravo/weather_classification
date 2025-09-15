# Menggunakan Image Dasar TensorFlow Serving Versi Terbaru
FROM tensorflow/serving:latest

# Mendefinisikan Variabel ARG untuk Nama Model
ARG MODEL_NAME=weather_classification

# Mendefinisikan Variabel ARG untuk Path Dasar Model
ARG MODEL_BASE_PATH=/models

# Menyalin Direktori Model dari Lokal ke Lokasi yang Ditentukan di Dalam Kontainer
COPY ./models/my_models/1757484741 ${MODEL_BASE_PATH}/${MODEL_NAME}/1

# Mengatur Variabel Lingkungan MODEL_NAME dengan Nilai yang Ditetapkan Sebelumnya
ENV MODEL_NAME=${MODEL_NAME}

# Mengatur Variabel Lingkungan MODEL_BASE_PATH dengan Nilai yang Ditetapkan Sebelumnya
ENV MODEL_BASE_PATH=${MODEL_BASE_PATH}

# Mengekspos Port Default 8500 untuk gRPC API
EXPOSE 8500

# Mengekspos Port Default 8501 untuk REST API
EXPOSE 8501

# Perintah Utama (Main Command) yang Dijalankan Saat Kontainer Dimulai
# Berfungsi untuk Menjalankan TensorFlow Model Server dan Mengonfigurasinya
CMD ["tensorflow_model_server", "--rest_api_port=8501", "--model_name=weather_classification", "--model_base_path=/models/weather_classification"]