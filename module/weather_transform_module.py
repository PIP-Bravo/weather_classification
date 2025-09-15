import tensorflow as tf # Mengimport Library TensorFlow untuk Melakukan Operasi Data dan Model
 
CLASS_NAMES = ["Cloudy", "Rain", "Shine", "Sunrise"] # Mendefinisikan Daftar Nama Kelas
IMG_SIZE = (128, 128) # Menentukan Ukuran Gambar yang Diinginkan
 
def preprocessing_fn(inputs): # Mendefinisikan Fungsi Pra-pemrosesan yang akan Dijalankan oleh Komponen Transform
    image_bytes = inputs['image'] # Mengambil Data Gambar dalam Format Bytes dari Input
    label_str = inputs['label'] # Mengambil Data Label dalam Format String dari Input
 
    def process_single_image(img_bytes): # Fungsi Pembantu untuk Memproses Satu Gambar Secara Independen
        img = tf.io.decode_image( # Mendekode Bytes Gambar Menjadi Tensor Gambar
            tf.reshape(img_bytes, []), # Memastikan Input Berbentuk Skalar (Scalar)
            channels=3, # Mengatur Gambar Agar Memiliki 3 Channel Warna (RGB)
            expand_animations=False # Memastikan Animasi (GIF) tidak Diperluas
        )
        img.set_shape([None, None, 3]) # Menetapkan Bentuk Tensor (Shape) Gambar
        img = tf.image.resize(img, IMG_SIZE) # Mengubah Ukuran Gambar menjadi IMG_SIZE
        img = tf.cast(img, tf.float32) / 255.0 # Mengubah Tipe Data Gambar Menjadi Float32 dan Menormalisasi Nilai Piksel ke Rentang 0-1
        return img # Mengembalikan Tensor Gambar yang Sudah Diproses
 
    images = tf.map_fn( # Menerapkan Fungsi process_single_image ke Seluruh Elemen di image_bytes
        process_single_image, # Fungsi yang akan Diterapkan pada Setiap Elemen
        image_bytes, # Tensor Input yang akan Dipetakan
        fn_output_signature=tf.TensorSpec(shape=(*IMG_SIZE, 3), dtype=tf.float32) # Mendefinisikan Tanda Tangan (Signature) Tensor Output
    )
 
    table = tf.lookup.StaticHashTable( # Membuat Hash Table untuk Mengubah Label String Menjadi Label Angka
        initializer=tf.lookup.KeyValueTensorInitializer( # Menginisialisasi Hash Table dengan Kunci dan Nilai
            keys=tf.constant(CLASS_NAMES), # Menggunakan Nama Kelas Sebagai Kunci
            values=tf.constant(list(range(len(CLASS_NAMES))), dtype=tf.int64), # Menggunakan Angka (Index) Sebagai Nilai
        ),
        default_value=-1 # Mengatur Nilai Default Jika Kunci Tidak Ditemukan
    )
    labels = table.lookup(label_str) # Mengubah Label String Menjadi Label Angka Menggunakan Hash Table
 
    return { # Mengembalikan Dictionary Berisi Fitur yang Telah Ditransformasi
        'image_xf': images, # Mengembalikan Tensor Gambar yang Telah Diproses
        'label_xf': labels # Mengembalikan Tensor Label yang Telah Ditransformasi Menjadi Angka
    }