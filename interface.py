# interface_code.py
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from proj import predict_sentiment  # Импорт функции из первого файла

# Создание основного окна
root = tk.Tk()
root.title("Анализатор отзывов")

# Темная тема окна
root.tk_setPalette(background='#2E2E2E', foreground='white')

# Виджет ввода текста
input_text_widget = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=40, height=10)
input_text_widget.grid(column=0, row=0, padx=10, pady=10, columnspan=2)

# Кнопка для анализа текста
def analyze_text():
    text = input_text_widget.get("1.0", tk.END)
    prediction = predict_sentiment(text)
    
    # Установка цвета текста в зависимости от результата
    if prediction == "Негативный":
        result_label.config(text=f"Тональность текста: {prediction}", foreground="red")
    elif prediction == "Положительный":
        result_label.config(text=f"Тональность текста: {prediction}", foreground="green")

analyze_button = ttk.Button(root, text="Анализировать", command=analyze_text)
analyze_button.grid(column=0, row=1, pady=10, columnspan=2)

# Метка для вывода результата
result_label = ttk.Label(root, text="")
result_label.grid(column=0, row=2, pady=10, columnspan=2)

# Запуск главного цикла обработки событий
root.mainloop()
