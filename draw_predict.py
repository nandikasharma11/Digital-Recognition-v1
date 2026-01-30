import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
from tensorflow import keras

model = keras.models.load_model("digit_model.h5")

canvas_size = 280
image_size = 28

root = tk.Tk()
root.title("Handwritten Digit Recognition")

canvas = tk.Canvas(root, width=280, height=280, bg="white", cursor="cross")
canvas.pack()
canvas.focus_set()

image = Image.new("L", (canvas_size, canvas_size), 255)
draw = ImageDraw.Draw(image)

def paint(event):
    x, y = event.x, event.y
    r = 16   # BIG brush so you can SEE it
    canvas.create_oval(
        x-r, y-r, x+r, y+r,
        fill="black", outline="black"
    )
    draw.ellipse([x-r, y-r, x+r, y+r], fill=0)
    

canvas.bind("<Button-1>", paint)
canvas.bind("<B1-Motion>", paint)

def predict():
    img = image.resize((28, 28))
    img = np.array(img)

    # VERY IMPORTANT: invert colors
    img = 255 - img

    img = img / 255.0
    img = img.reshape(1, 28, 28)

    pred = model.predict(img)
    digit = np.argmax(pred)

    result.config(text=f"Predicted Digit: {digit}")

def clear():
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_size, canvas_size], fill=255)
    result.config(text="Draw a digit")

tk.Button(root, text="Predict", command=predict).pack()
tk.Button(root, text="Clear", command=clear).pack()

result = tk.Label(
    root,
    text="Draw a digit",
    font=("Arial", 18, "bold"),
    fg="blue"
)
result.pack(pady=10)

root.mainloop()
