import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# 设置数据集路径
data_dir = '/Users/lusicheng/Desktop/summerresearch2024/VGG16/trainingdata/hs'

# 加载数据集
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2, # 从训练集中拿出20%作为验证集
    subset="training", # 设置加载训练集
    seed=123, # 设置随机种子方便复现
    image_size=(150, 150), # 设置输入图像尺寸
    batch_size=32 # 设置批大小
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation", # 设置加载验证集
    seed=123,
    image_size=(150, 150),
    batch_size=32
)

# 获取类别数量
num_classes = len(train_ds.class_names)

# 数据预处理
train_ds = train_ds.map(lambda x, y: (preprocess_input(x), to_categorical(y, num_classes=num_classes)))
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), to_categorical(y, num_classes=num_classes)))

# 加载VGG16模型
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False

# 模型架构
flatten_layer = layers.Flatten()
dense_layer_1 = layers.Dense(50, activation='relu')
dense_layer_2 = layers.Dense(20, activation='relu')
prediction_layer = layers.Dense(num_classes, activation='softmax')

model = models.Sequential([
    base_model,
    flatten_layer,
    dense_layer_1,
    dense_layer_2,
    prediction_layer
])

# 模型编译
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 早停法
es = EarlyStopping(monitor='val_accuracy', mode='max', patience=5, restore_best_weights=True)

# 训练模型
history = model.fit(train_ds, epochs=50, validation_data=val_ds, callbacks=[es])

# 保存模型
model.save('/Users/lusicheng/Desktop/summerresearch2024/VGG16/VGG16_train/modelhuasi.h5')
