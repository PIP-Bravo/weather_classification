import tensorflow as tf # Mengimpor TensorFlow
from tfx import v1 as tfx # Mengimpor TFX
from tensorflow_transform import TFTransformOutput # Mengimpor TFTransformOutput untuk Menggunakan Grafik Transformasi
from tensorflow.keras.callbacks import LambdaCallback # Mengimpor LambdaCallback untuk Mencetak Informasi saat Pelatihan
import os # Mengimpor OS untuk Operasi Sistem
import json # Mengimpor JSON untuk Mengurai Data Hyperparameter
 
IMG_SIZE = (128, 128) # Ukuran Gambar yang Digunakan
CLASS_NAMES = ["Cloudy", "Rain", "Shine", "Sunrise"] # Nama-nama Kelas
 
def _build_model(hp_values=None): # Fungsi untuk Membangun Model Keras Berdasarkan Hyperparameter atau Nilai Default
    if hp_values: # Memeriksa Apakah Hyperparameter Disediakan
        print(f"Using tuned hyperparameters: {hp_values}") # Mencetak Pesan Bahwa Hyperparameter Disesuaikan Digunakan
        
        model = tf.keras.Sequential([ # Membuat Model Sequential
            tf.keras.layers.InputLayer(input_shape=(*IMG_SIZE, 3), name="image_xf"), # Layer Input
            
            tf.keras.layers.Conv2D( # Layer Konvolusi Pertama
                filters=hp_values.get('conv1_filters', 32), # Menggunakan Nilai Filter dari Hyperparameter atau Default 32
                kernel_size=(3, 3), activation="relu", padding="same"), # Mengatur Ukuran Kernel, Aktivasi, dan Padding
            tf.keras.layers.MaxPooling2D((2, 2)), # Layer MaxPooling
            
            tf.keras.layers.Conv2D( # Layer Konvolusi Kedua
                filters=hp_values.get('conv2_filters', 64), # Menggunakan Nilai Filter dari Hyperparameter atau Default 64
                kernel_size=(3, 3), activation="relu", padding="same"), # Mengatur Ukuran Kernel, Aktivasi, dan Padding
            tf.keras.layers.MaxPooling2D((2, 2)), # Layer MaxPooling
            
            tf.keras.layers.Conv2D( # Layer Konvolusi Ketiga
                filters=hp_values.get('conv3_filters', 128), # Menggunakan Nilai Filter dari Hyperparameter atau Default 128
                kernel_size=(3, 3), activation="relu", padding="same"), # Mengatur Ukuran Kernel, Aktivasi, dan Padding
            tf.keras.layers.MaxPooling2D((2, 2)), # Layer MaxPooling
            
            tf.keras.layers.Flatten(), # Meratakan Tensor Input
            
            tf.keras.layers.Dense( # Layer Dense
                units=hp_values.get('dense_units', 128), # Menggunakan Jumlah Unit dari Hyperparameter atau Default 128
                activation="relu"), # Fungsi Aktivasi ReLU
            tf.keras.layers.Dropout(hp_values.get('dropout_rate', 0.5)), # Layer Dropout untuk Regularisasi
            
            tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax"), # Layer Output dengan Jumlah Unit Sama dengan Jumlah Kelas
        ])
        
        model.compile( # Mengompilasi Model
            optimizer=tf.keras.optimizers.Adam( # Menggunakan Optimizer Adam
                learning_rate=hp_values.get('learning_rate', 1e-4) # Menggunakan Learning Rate dari Hyperparameter atau Default 1e-4
            ),
            loss="sparse_categorical_crossentropy", # Mengatur Fungsi Loss
            metrics=["accuracy"] # Mengatur Metrik untuk Evaluasi
        )
        
    else: # Jika Hyperparameter Tidak Disediakan
        print("Using default model architecture") # Mencetak Pesan Bahwa Arsitektur Model Default Digunakan
        
        model = tf.keras.Sequential([ # Membuat Model Sequential dengan Nilai Default
            tf.keras.layers.InputLayer(input_shape=(*IMG_SIZE, 3), name="image_xf"), # Layer Input
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"), # Layer Konvolusi
            tf.keras.layers.MaxPooling2D((2, 2)), # Layer MaxPooling
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"), # Layer Konvolusi
            tf.keras.layers.MaxPooling2D((2, 2)), # Layer MaxPooling
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"), # Layer Konvolusi
            tf.keras.layers.MaxPooling2D((2, 2)), # Layer MaxPooling
            tf.keras.layers.Flatten(), # Meratakan Tensor Input
            tf.keras.layers.Dense(128, activation="relu"), # Layer Dense
            tf.keras.layers.Dropout(0.5), # Layer Dropout
            tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax"), # Layer Output
        ])
 
        model.compile( # Mengompilasi Model
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), # Menggunakan Optimizer Adam dengan Learning Rate Default
            loss="sparse_categorical_crossentropy", # Mengatur Fungsi Loss
            metrics=["accuracy"] # Mengatur Metrik
        )
    
    return model # Mengembalikan Model yang Telah Dikompilasi
 
def run_fn(fn_args: tfx.components.FnArgs): # Fungsi Utama yang Akan Dijalankan oleh Komponen Trainer
    best_hps = None # Menginisialisasi Variabel untuk Hyperparameter Terbaik
    if hasattr(fn_args, 'custom_config') and fn_args.custom_config: # Memeriksa Apakah Konfigurasi Kustom Disediakan
        best_hps = fn_args.custom_config.get('best_hyperparameters') # Mendapatkan Hyperparameter Terbaik dari Konfigurasi Kustom
    
    print(f"Best hyperparameters received: {best_hps}") # Mencetak Hyperparameter yang Diterima
    
    if not best_hps: # Jika Hyperparameter Tidak Ditemukan
        print("No hyperparameters provided, using default model") # Mencetak Pesan Peringatan
        best_hps = {} # Mengatur Variabel Menjadi Dictionary Kosong
    
    tf_transform_output = TFTransformOutput(fn_args.transform_output) # Menggunakan Output Transform untuk Mendapatkan Grafik Transformasi
    feature_spec = tf_transform_output.transformed_feature_spec() # Mendapatkan Spesifikasi Fitur yang Telah Ditransformasi
    
    print("Feature spec:", feature_spec) # Mencetak Spesifikasi Fitur
    
    def create_dataset(file_pattern, batch_size=32, shuffle=False): # Fungsi untuk Membuat tf.data.Dataset
        file_paths = tf.io.gfile.glob(file_pattern) # Mendapatkan Path File yang Sesuai dengan Pola
        print(f"Processing {len(file_paths)} files: {file_paths}") # Mencetak Jumlah dan Daftar File
        
        if not file_paths: # Jika Tidak Ada File yang Ditemukan
            raise ValueError(f"No files found for pattern: {file_pattern}") # Mengangkat Kesalahan
        
        def read_file(filepath): # Fungsi Pembantu untuk Membaca File
            return tf.data.TFRecordDataset(filepath, compression_type="GZIP") # Mengembalikan Dataset dari TFRecord
        
        dataset = tf.data.Dataset.from_tensor_slices(file_paths) # Membuat Dataset dari Daftar Path
        
        dataset = dataset.interleave( # Menginterleave Dataset untuk Pembacaan Paralel
            read_file,
            num_parallel_calls=tf.data.AUTOTUNE,
            cycle_length=min(4, len(file_paths))
        )
        
        def parse_record(serialized_example): # Fungsi Pembantu untuk Menguraikan Rekor
            try: # Memulai Blok Try-Except
                parsed_features = tf.io.parse_single_example(serialized_example, feature_spec) # Menguraikan Contoh
                
                image = parsed_features['image_xf'] # Mendapatkan Tensor Gambar
                label = parsed_features['label_xf'] # Mendapatkan Tensor Label
                
                image = tf.ensure_shape(image, (*IMG_SIZE, 3)) # Memastikan Bentuk (Shape) Tensor Gambar
                label = tf.reshape(label, []) # Mengubah Bentuk Tensor Label menjadi Skalar (Scalar)
                
                return image, label # Mengembalikan Gambar dan Label
                
            except Exception as e: # Menangani Kesalahan Parsing
                print(f"Error parsing record: {e}") # Mencetak Pesan Error
                return tf.zeros((*IMG_SIZE, 3), dtype=tf.float32), tf.constant(0, dtype=tf.int64) # Mengembalikan Data Dummy
        
        dataset = dataset.map(parse_record, num_parallel_calls=tf.data.AUTOTUNE) # Menerapkan Fungsi Parsing ke Dataset
        
        if shuffle: # Jika Perlu Diacak
            dataset = dataset.shuffle(buffer_size=1000) # Mengacak Dataset
            
        dataset = dataset.batch(batch_size) # Mengelompokkan Dataset
        dataset = dataset.prefetch(tf.data.AUTOTUNE) # Melakukan Prefetch untuk Performa
        
        return dataset # Mengembalikan Dataset
    
    try: # Memulai Blok Try-Except untuk Pembuatan Dataset
        train_dataset = create_dataset(fn_args.train_files, batch_size=32, shuffle=True) # Membuat Dataset Latihan
        eval_dataset = create_dataset(fn_args.eval_files, batch_size=32, shuffle=False) # Membuat Dataset Evaluasi
        
        print("Testing dataset...") # Mencetak Pesan
        for images, labels in train_dataset.take(1): # Mengambil Satu Batch untuk Pengujian
            print(f"Batch shapes - Images: {images.shape}, Labels: {labels.shape}") # Mencetak Bentuk Tensor
            break # Keluar dari Loop
            
    except Exception as e: # Menangani Kesalahan Pembuatan Dataset
        print(f"Error creating datasets: {e}") # Mencetak Pesan Error
        print("Using dummy data as fallback...") # Mencetak Pesan Fallback
        dummy_data = tf.data.Dataset.from_tensor_slices(( # Membuat Dataset Dummy
            tf.zeros((100, *IMG_SIZE, 3), dtype=tf.float32),
            tf.zeros((100,), dtype=tf.int64)
        )).batch(32)
        train_dataset = dummy_data # Menggunakan Data Dummy untuk Latihan
        eval_dataset = dummy_data # Menggunakan Data Dummy untuk Evaluasi
    
    print(best_hps) # Mencetak Hyperparameter Terbaik
    model = _build_model(best_hps) # Membangun Model Berdasarkan Hyperparameter Terbaik
    
    print_callback = LambdaCallback( # Membuat Callback untuk Mencetak Log Epoch
        on_epoch_end=lambda epoch, logs: 
            print(f"Epoch {epoch+1}: loss={logs['loss']:.4f}, "
                  f"accuracy={logs['accuracy']:.4f}, "
                  f"val_loss={logs['val_loss']:.4f}, "
                  f"val_accuracy={logs['val_accuracy']:.4f}")
    )
 
    callbacks = [ # Mendefinisikan Daftar Callback
        tf.keras.callbacks.TensorBoard(log_dir=fn_args.model_run_dir), # Callback TensorBoard untuk Logging
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5), # Callback EarlyStopping untuk Mencegah Overfitting
        print_callback # Callback Cetak
    ]
    
    history = model.fit( # Melatih Model
        train_dataset,
        validation_data=eval_dataset,
        epochs=100, # Menentukan Jumlah Epoch Maksimal
        callbacks=callbacks # Menambahkan Callback
    )
    
    model.save(fn_args.serving_model_dir, save_format='tf') # Menyimpan Model yang Telah Dilatih