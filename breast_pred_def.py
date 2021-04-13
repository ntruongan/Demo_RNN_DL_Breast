def breast_pred_result(input_path,model_path):
    import numpy as np
    import tensorflow as tf
    from keras.preprocessing import image
    test_image = image.load_img(input_path, target_size=(300, 300)) #input_path here
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    model = tf.keras.models.load_model(model_path) #model_path here
    result = model.predict(test_image)
    if (max(result[0][0], result[0][1], result[0][2], result[0][3]) == result[0][0]):
        print('Benign')
    elif (max(result[0][0], result[0][1], result[0][2], result[0][3]) == result[0][1]):
        print('inSitu')
    elif (max(result[0][0], result[0][1], result[0][2], result[0][3]) == result[0][2]):
        print('Invasive')
    else:
        print('Normal')

breast_pred_result('BreastDataset/Normal/N (20).tif','breast_classifier.h5')