import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import proj as p


# Создание основного окна
root = tk.Tk()
root.title("Анализатор отзывов")

# Виджет ввода текста
input_text_widget = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=40, height=10)
input_text_widget.grid(column=0, row=0, padx=10, pady=10, columnspan=2)

# Кнопка для анализа текста
analyze_button = ttk.Button(root, text="Анализировать", command=p.analyze_text)
analyze_button.grid(column=0, row=1, pady=10, columnspan=2)

# Метка для вывода результата
result_label = ttk.Label(root, text="")
result_label.grid(column=0, row=2, pady=10, columnspan=2)

# Запуск главного цикла обработки событий
root.mainloop()