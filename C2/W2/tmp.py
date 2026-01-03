import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def preprocess_image(image_path, show_image=False):
    """
    预处理手写数字图片，使其与训练数据格式一致
    Args:
        image_path: 图片路径
        show_image: 是否显示处理前后的图片
    Returns:
        processed_image: 处理后的400维向量
    """
    # 1. 读取图片
    # 使用OpenCV读取
    img_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 如果OpenCV读取失败，尝试用PIL
    if img_cv is None:
        img_pil = Image.open(image_path).convert('L')
        img_array = np.array(img_pil)
    else:
        img_array = img_cv
    
    if show_image:
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(img_array, cmap='gray')
        plt.title('Original Image')
    
    # 2. 调整大小为20x20像素（与训练数据一致）
    img_resized = cv2.resize(img_array, (20, 20))
    
    # 3. 将像素值归一化到0-1之间
    # 注意：如果图片是白底黑字，需要反转为黑底白字（与训练数据一致）
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # 4. 反转颜色（如果需要的话，使数字为白色，背景为黑色）
    # 检测是否需要反转：如果大部分像素是亮的（白底），则需要反转
    if np.mean(img_normalized) > 0.5:
        img_normalized = 1.0 - img_normalized
    
    # 5. 确保对比度足够
    # 可以通过重新缩放像素值来增强对比度
    if np.max(img_normalized) - np.min(img_normalized) < 0.5:
        img_normalized = (img_normalized - np.min(img_normalized)) / (np.max(img_normalized) - np.min(img_normalized) + 1e-8)
    
    if show_image:
        plt.subplot(1, 2, 2)
        plt.imshow(img_normalized, cmap='gray')
        plt.title('Processed Image (20x20)')
        plt.show()
    
    # 6. 展平为400维向量
    img_flattened = img_normalized.flatten()
    
    return img_flattened.reshape(1, 400)  # 添加batch维度

def predict_digit(image_path, model_path="model/multiclass_handwritten.keras", show_image=True):
    """
    预测手写数字
    Args:
        image_path: 图片路径
        model_path: 模型路径
        show_image: 是否显示图片
    Returns:
        prediction: 预测结果
        probabilities: 概率分布
    """
    # 1. 加载模型
    print("加载模型...")
    model = tf.keras.models.load_model(model_path)
    print("模型加载完成！")
    
    # 2. 预处理图片
    print("预处理图片...")
    processed_image = preprocess_image(image_path, show_image=show_image)
    
    # 3. 进行预测
    print("进行预测...")
    logits = model.predict(processed_image, verbose=0)
    
    # 4. 应用softmax得到概率
    probabilities = tf.nn.softmax(logits).numpy()[0]
    
    # 5. 获取预测结果
    predicted_digit = np.argmax(probabilities)
    confidence = probabilities[predicted_digit]
    
    return predicted_digit, probabilities, confidence, processed_image

def display_prediction_results(image_path, predicted_digit, probabilities, confidence):
    """
    显示预测结果
    """
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 显示图片
    img = Image.open(image_path).convert('L')
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(f'输入图片\n预测数字: {predicted_digit} (置信度: {confidence:.2%})')
    axes[0].axis('off')
    
    # 显示概率分布
    x_pos = np.arange(10)
    bars = axes[1].bar(x_pos, probabilities, color='skyblue')
    
    # 高亮显示预测的数字
    bars[predicted_digit].set_color('red')
    
    axes[1].set_xlabel('数字')
    axes[1].set_ylabel('概率')
    axes[1].set_title('预测概率分布')
    axes[1].set_xticks(x_pos)
    axes[1].set_ylim([0, 1])
    
    # 在柱状图上显示概率值
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.2%}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # 打印详细结果
    print("\n" + "="*50)
    print(f"预测结果: 数字 {predicted_digit}")
    print(f"置信度: {confidence:.2%}")
    print("-"*50)
    print("所有数字的概率分布:")
    for i, prob in enumerate(probabilities):
        if i == predicted_digit:
            print(f"  数字 {i}: {prob:.2%}  ← 预测结果")
        else:
            print(f"  数字 {i}: {prob:.2%}")

# 使用示例
if __name__ == "__main__":
    # 示例1: 使用你自己的手写数字图片
    image_path = "path/to/your/handwritten_digit.jpg"  # 替换为你的图片路径
    
    # 进行预测
    predicted_digit, probabilities, confidence, _ = predict_digit(
        image_path, 
        model_path="model/multiclass_handwritten.keras",
        show_image=True
    )
    
    # 显示结果
    display_prediction_results(image_path, predicted_digit, probabilities, confidence)