from keras.layers import *
from keras.layers import AveragePooling2D
from keras.models import Model

def generator():
    noise = Input(shape=(25,))
    x = Dense(8*8*128,activation = 'relu')(noise)
    x = Reshape((8,8,128))(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Conv2DTranspose(129,2,strides=2,padding='valid')(x)

    x = Conv2D(128,3,padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2D(128,3,padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(128,2,strides=2,padding='valid')(x)
    x = Conv2D(128,3,padding='same')(x)
    x = BatchNormalization(momentum = 0.8)(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(128,3,padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2D(128,3,padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(128,2,strides=2,padding='valid')(x)
    x = Conv2D(128,3,padding='same')(x)
    x = BatchNormalization(momentum = 0.8)(x)
    x = LeakyReLU()(x)

    x = Conv2D(128,3,padding='same')(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(64,3,padding='same')(x)
    x = LeakyReLU()(x)
    
    out = Conv2D(3,3,padding='same',activation = 'sigmoid')(x)

    
    return Model(noise,out,name = 'generator')
    

def discriminator():
    image = Input(shape=(64,64,3))

    x = Conv2D(64,3,padding='same')(image)
    x = LeakyReLU()(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(64,3,strides=2,padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.25)(x)

    
    x1 = Conv2D(64,3,strides=2,padding='same')(x)
    x1 = LeakyReLU()(x1)
    x1 = Dropout(0.25)(x1)

    x2 = Conv2D(64,5,strides=2,padding='same')(x)
    x2 = LeakyReLU()(x2)
    x2 = Dropout(0.25)(x2)

    x = Concatenate()([x1,x2])

    x = Conv2D(128,3,strides=2,padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.25)(x)    

    x = Conv2D(128,3,padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.25)(x)


    x = Flatten()(x)
    x = Dense(32)(x)
    x = LeakyReLU()(x)
    out = Dense(1,activation = 'sigmoid')(x)

    return Model(image,out,name = 'discriminator')

def generator128():
    noise = Input(shape=(25,))
    x = Dense(8*8*128,activation = 'relu')(noise)
    x = Reshape((8,8,128))(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Conv2DTranspose(128,2,strides=2,padding='valid')(x)

    x = Conv2D(128,3,padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2D(128,3,padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(128,2,strides=2,padding='valid')(x)
    x = Conv2D(128,3,padding='same')(x)
    x = BatchNormalization(momentum = 0.8)(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(128,3,padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2D(128,3,padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(128,2,strides=2,padding='valid')(x)
    x = Conv2D(128,3,padding='same')(x)
    x = BatchNormalization(momentum = 0.8)(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(128,3,padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2D(128,3,padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(128,2,strides=2,padding='valid')(x)
    x = Conv2D(128,3,padding='same')(x)
    x = BatchNormalization(momentum = 0.8)(x)
    x = LeakyReLU()(x)

    x = Conv2D(128,3,padding='same')(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(64,3,padding='same')(x)
    x = LeakyReLU()(x)
    
    out = Conv2D(3,3,padding='same',activation = 'sigmoid')(x)

    
    return Model(noise,out,name = 'generator')
    

def discriminator128():
    image = Input(shape=(128,128,3))

    x = Conv2D(64,3,padding='same')(image)
    x = LeakyReLU()(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(64,3,strides=2,padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.25)(x)

    
    x1 = Conv2D(64,3,strides=2,padding='same')(x)
    x1 = LeakyReLU()(x1)
    x1 = Dropout(0.25)(x1)

    x2 = Conv2D(64,5,strides=2,padding='same')(x)
    x2 = LeakyReLU()(x2)
    x2 = Dropout(0.25)(x2)

    x = Concatenate()([x1,x2])
    x = Conv2D(64,3,padding='same')(x)

    x1 = Conv2D(64,3,strides=2,padding='same')(x)
    x1 = LeakyReLU()(x1)
    x1 = Dropout(0.25)(x1)

    x2 = Conv2D(64,5,strides=2,padding='same')(x)
    x2 = LeakyReLU()(x2)
    x2 = Dropout(0.25)(x2)

    x = Concatenate()([x1,x2])

    x = Conv2D(128,3,strides=2,padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.25)(x)    

    x = Conv2D(128,3,padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.25)(x)


    x = Flatten()(x)
    x = Dense(32)(x)
    x = LeakyReLU()(x)
    out = Dense(1,activation = 'sigmoid')(x)

    return Model(image,out,name = 'discriminator')

def conditionalgen():
    noise = Input(shape=(25,))
    conditions = Input(shape=(40,))

    x1 = Dense(8*8*64,activation='relu')(noise)
    x1 = Reshape((8,8,64))(x1)
    x2 = Dense(8*8*64,activation='relu')(conditions)
    x2 = Reshape((8,8,64))(x2)
    x = Concatenate()([x1,x2])
    x = BatchNormalization(momentum=0.8)(x)

    x = Conv2DTranspose(128,2,strides=2,padding='valid')(x)
    x = Convolution2D(128,3,padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU()(x)

    x1 = Conv2D(64,3,padding='same')(x)
    x2 = Conv2D(64,5,padding='same')(x)
    x = Concatenate()([x1,x2])
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(128,2,strides=2,padding='valid')(x)
    x = Convolution2D(128,3,padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU()(x)

    x1 = Conv2D(64,3,padding='same')(x)
    x2 = Conv2D(64,5,padding='same')(x)
    x = Concatenate()([x1,x2])
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(128,2,strides=2,padding='valid')(x)
    x = Convolution2D(128,3,padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU()(x)

    x1 = Conv2D(64,3,padding='same')(x)
    x2 = Conv2D(64,5,padding='same')(x)
    x = Concatenate()([x1,x2])
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU()(x)

    x = Conv2D(128,3,padding='same')(x)
    x = Conv2D(64,3,padding='same')(x)
    out = Conv2D(3,5,padding='same')(x)

    model = Model([noise,conditions],out,name='condgen')
    model.summary()
    return model


def conditionaldiscriminator():
    image = Input(shape=(64,64,3))

