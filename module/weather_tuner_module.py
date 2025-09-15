import tensorflow as tf # Mengimpor TensorFlow untuk Membuat dan Melatih Model
import kerastuner as kt # Mengimpor KerasTuner untuk Penyetelan Hyperparameter
from tfx import v1 as tfx # Mengimpor TFX untuk Komponen dan Jenis Data
from tensorflow_transform import TFTransformOutput # Mengimpor TFTransformOutput untuk Mengakses Output Transform
import os # Mengimpor OS untuk Operasi Sistem File
from tfx.components.tuner.component import TunerFnResult # Mengimpor TunerFnResult untuk Mengembalikan Hasil dari tuner_fn
 
IMG_SIZE = (128, 128) # Menentukan Ukuran Standar untuk Semua Gambar Input
CLASS_NAMES = ["Cloudy", "Rain", "Shine", "Sunrise"] # Mendefinisikan Nama Kelas Kategori
 
def create_dataset(file_pattern, feature_spec, batch_size=32, shuffle=False): # Fungsi untuk Membuat tf.data.Dataset dari File TFRecord
    file_paths = tf.io.gfile.glob(file_pattern) # Mendapatkan Daftar Semua Path File yang Sesuai dengan Pola
    if not file_paths: # Memeriksa Apakah Ada File yang Ditemukan
        raise ValueError(f"No files found for pattern: {file_pattern}") # Mengangkat Kesalahan Jika Tidak Ada File yang Ditemukan
 
    dataset = tf.data.Dataset.from_tensor_slices(file_paths) # Membuat Dataset dari Daftar Path File
    dataset = dataset.interleave( # Memuat dan Memproses Data dari Berbagai File Secara Paralel
        lambda filepath: tf.data.TFRecordDataset(filepath, compression_type="GZIP"), # Fungsi untuk Membaca File TFRecord
        num_parallel_calls=tf.data.AUTOTUNE, # Menggunakan Panggilan Paralel Otomatis untuk Pemuatan Data yang Lebih Cepat
        cycle_length=min(4, len(file_paths)) # Menentukan Jumlah File yang akan Diproses Secara Bersamaan
    )
 
    def parse_record(serialized_example): # Fungsi Pembantu untuk Menguraikan Contoh Berseri dari TFRecord
        parsed_features = tf.io.parse_single_example(serialized_example, feature_spec) # Menguraikan Contoh ke Fitur-Fitur
        image = parsed_features['image_xf'] # Mengambil Fitur Gambar yang Telah Ditransformasi
        label = parsed_features['label_xf'] # Mengambil Fitur Label yang Telah Ditransformasi
        image = tf.ensure_shape(image, (*IMG_SIZE, 3)) # Memastikan Bentuk (Shape) Tensor Gambar
        label = tf.reshape(label, []) # Mengubah Bentuk Tensor Label menjadi Skalar (Scalar)
        return image, label # Mengembalikan Pasangan Gambar dan Label
 
    dataset = dataset.map(parse_record, num_parallel_calls=tf.data.AUTOTUNE) # Menerapkan Fungsi Parsing ke Setiap Rekor Secara Paralel
    if shuffle: # Memeriksa Apakah Data Perlu Diacak
        dataset = dataset.shuffle(buffer_size=1000) # Mengacak Dataset dengan Buffer yang Ditetapkan
 
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE) # Mengelompokkan Data dan Melakukan Prefetch untuk Performa
    return dataset # Mengembalikan Dataset yang Siap untuk Digunakan
 
def _build_keras_model(hp): # Fungsi untuk Membangun Model Keras dengan Hyperparameter yang Dapat Disetel
    model = tf.keras.Sequential([ # Membuat Model Sequential
        tf.keras.layers.InputLayer(input_shape=(*IMG_SIZE, 3), name="image_xf"), # Menambahkan Layer Input
 
        tf.keras.layers.Conv2D( # Layer Konvolusi Pertama
            filters=hp.Int("conv1_filters", 32, 64, step=16), # Menyetel Jumlah Filter
            kernel_size=(3, 3), activation="relu", padding="same"), # Menentukan Ukuran Kernel, Fungsi Aktivasi, dan Padding
        tf.keras.layers.MaxPooling2D((2, 2)), # Layer MaxPooling Pertama
 
        tf.keras.layers.Conv2D( # Layer Konvolusi Kedua
            filters=hp.Int("conv2_filters", 64, 128, step=32), # Menyetel Jumlah Filter
            kernel_size=(3, 3), activation="relu", padding="same"), # Menentukan Ukuran Kernel, Fungsi Aktivasi, dan Padding
        tf.keras.layers.MaxPooling2D((2, 2)), # Layer MaxPooling Kedua
 
        tf.keras.layers.Conv2D( # Layer Konvolusi Ketiga
            filters=hp.Int("conv3_filters", 128, 256, step=64), # Menyetel Jumlah Filter
            kernel_size=(3, 3), activation="relu", padding="same"), # Menentukan Ukuran Kernel, Fungsi Aktivasi, dan Padding
        tf.keras.layers.MaxPooling2D((2, 2)), # Layer MaxPooling Ketiga
 
        tf.keras.layers.Flatten(), # Meratakan Output dari Layer Sebelumnya
 
        tf.keras.layers.Dense( # Layer Dense Pertama
            units=hp.Int("dense_units", 64, 256, step=64), # Menyetel Jumlah Unit
            activation="relu"), # Menentukan Fungsi Aktivasi
        tf.keras.layers.Dropout(hp.Float("dropout_rate", 0.3, 0.5, step=0.1)), # Layer Dropout untuk Regularisasi
 
        tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax"), # Layer Output dengan Jumlah Unit Sama dengan Jumlah Kelas
    ])
 
    lr = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4]) # Menyetel Learning Rate
    model.compile( # Mengompilasi Model
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr), # Menggunakan Optimizer Adam dengan Learning Rate yang Dapat Disetel
        loss="sparse_categorical_crossentropy", # Menentukan Fungsi Loss untuk Klasifikasi
        metrics=["accuracy"] # Menentukan Metrik untuk Evaluasi
    )
    return model # Mengembalikan Model yang Telah Dikompilasi
 
def tuner_fn(fn_args: tfx.components.FnArgs): # Fungsi Penyetelan Utama yang Dijalankan oleh Komponen Tuner
    print(">>> Debug: Starting tuner_fn") # Mencetak Pesan Debug
    
    if not hasattr(fn_args, 'custom_config') or not fn_args.custom_config: # Memeriksa Keberadaan Konfigurasi Kustom
        raise ValueError("custom_config is required for accessing transform graph") # Mengangkat Kesalahan Jika Konfigurasi Kustom Tidak Ada
    
    transform_graph_uri = fn_args.custom_config.get('transform_graph_uri') # Mendapatkan URI dari Grafik Transform
    if not transform_graph_uri: # Memeriksa Keberadaan URI
        raise ValueError("transform_graph_uri not found in custom_config") # Mengangkat Kesalahan Jika URI Tidak Ditemukan
    
    print(f"Using transform graph from: {transform_graph_uri}") # Mencetak URI Grafik Transform
    
    if not tf.io.gfile.exists(transform_graph_uri): # Memeriksa Apakah Path Grafik Transform Benar-Benar Ada
        raise ValueError(f"Transform graph path does not exist: {transform_graph_uri}") # Mengangkat Kesalahan Jika Path Tidak Ditemukan
    
    tf_transform_output = TFTransformOutput(transform_graph_uri) # Memuat Output Transform dari URI
    feature_spec = tf_transform_output.transformed_feature_spec() # Mendapatkan Spesifikasi Fitur yang Telah Ditransformasi
    print("Feature spec loaded successfully") # Mencetak Pesan Berhasil
 
    train_dataset = create_dataset(fn_args.train_files, feature_spec, batch_size=32, shuffle=True) # Membuat Dataset Latihan
    eval_dataset = create_dataset(fn_args.eval_files, feature_spec, batch_size=32, shuffle=False) # Membuat Dataset Evaluasi
 
    train_dataset = train_dataset.repeat() # Mengatur Dataset Latihan agar Berulang Tanpa Batas
    eval_dataset = eval_dataset.repeat() # Mengatur Dataset Evaluasi agar Berulang Tanpa Batas
 
    def count_examples(file_pattern): # Fungsi Pembantu untuk Menghitung Jumlah Contoh dalam File
        file_paths = tf.io.gfile.glob(file_pattern) # Mendapatkan Path File
        count = 0 # Menginisialisasi Penghitung
        for file_path in file_paths: # Iterasi Melalui Setiap File
            dataset = tf.data.TFRecordDataset(file_path, compression_type="GZIP") # Membaca Dataset TFRecord
            count += sum(1 for _ in dataset) # Menambahkan Jumlah Contoh ke Penghitung
        return count # Mengembalikan Total Jumlah Contoh
 
    try: # Blok Try-Except untuk Mengatasi Potensi Error
        train_examples = count_examples(fn_args.train_files) # Menghitung Jumlah Contoh Latihan
        eval_examples = count_examples(fn_args.eval_files) # Menghitung Jumlah Contoh Evaluasi
        
        train_steps = train_examples // 32 # Menghitung Jumlah Langkah Latihan Per Epoch
        eval_steps = eval_examples // 32 # Menghitung Jumlah Langkah Evaluasi Per Epoch
        
        print(f"Training examples: {train_examples}, Steps: {train_steps}") # Mencetak Jumlah Contoh dan Langkah Latihan
        print(f"Eval examples: {eval_examples}, Steps: {eval_steps}") # Mencetak Jumlah Contoh dan Langkah Evaluasi
        
    except Exception as e: # Menangkap Pengecualian Jika Terjadi Error
        print(f"Error counting examples: {e}") # Mencetak Pesan Error
        train_steps = fn_args.train_steps # Menggunakan Langkah Default Jika Perhitungan Gagal
        eval_steps = fn_args.eval_steps # Menggunakan Langkah Default Jika Perhitungan Gagal
 
    tuner = kt.RandomSearch( # Menginisialisasi Tuner dengan Algoritma Random Search
        hypermodel=_build_keras_model, # Menentukan Model Keras yang akan Disesuaikan
        objective="val_accuracy", # Mendefinisikan Tujuan Penyetelan (Memaksimalkan Akurasi Validasi)
        max_trials=3, # Menentukan Jumlah Percobaan Maksimal
        executions_per_trial=1, # Menentukan Jumlah Eksekusi Model per Percobaan
        directory=fn_args.working_dir, # Menentukan Direktori Kerja untuk Tuner
        project_name="weather_tuner" # Menentukan Nama Proyek Tuner
    )
 
    return TunerFnResult( # Mengembalikan Hasil dari tuner_fn
        tuner=tuner, # Mengembalikan Objek Tuner
        fit_kwargs={ # Mengembalikan Argumen untuk Fungsi fit() Model
            "x": train_dataset, # Data Latihan
            "validation_data": eval_dataset, # Data Validasi
            "steps_per_epoch": train_steps, # Langkah-Langkah per Epoch untuk Latihan
            "validation_steps": eval_steps, # Langkah-Langkah per Epoch untuk Validasi
            "epochs": 10 # Jumlah Epoch untuk Latihan
        }
    )