from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.graphics import Color, Ellipse, Line
import pandas as pd

class MyPaintWidget(Widget):
    data = []

    def on_touch_down(self, touch):
        # color = (random(), 1, 1)
        color = (100, 100, 100)
        with self.canvas:
            Color(*color, mode='hsv')
            d = 10.
            Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d))
            touch.ud['line'] = Line(points=(touch.x, touch.y))
            self.data.append([touch.x, touch.y])

    def on_touch_move(self, touch):
        touch.ud['line'].points += [touch.x, touch.y]
        self.data.append([touch.x, touch.y])
        
    def on_touch_up(self, touch):
        # print(self.data)
        pass
        


class MyPaintApp(App):
    def build(self):
        parent = Widget()
        self.painter = MyPaintWidget()
        parent.add_widget(self.painter)

        clearbtn = Button(text='Clear')
        clearbtn.bind(on_release=self.clear_canvas)
        parent.add_widget(clearbtn)

        savebtn = Button(text='Save', pos=(parent.x, parent.y+100))
        savebtn.bind(on_release=self.save_data)
        parent.add_widget(savebtn)

        self.nameText = TextInput(text='', multiline=False, font_size = 30, size_hint_y = None, height = 50, pos=(parent.x+100, parent.y))
        parent.add_widget(self.nameText)

        return parent

    def clear_canvas(self, obj):
        self.painter.canvas.clear()
        self.painter.data = []

    def save_data(self, obj):
        if len(self.nameText.text) == 0: nameText = "NoName"
        else: nameText = self.nameText.text
        if len(self.painter.data) > 5:
            df = pd.DataFrame(self.painter.data, columns=['x', 'y'])
            df.to_csv(path_or_buf="paint_output/" + nameText + ".csv", header=True)
            print("saved")
        else: print("Not saved. Data points less than 5 cannot be saved.")

    def get_text(self, text):
        print(text)
        return text


if __name__ == '__main__':
    MyPaintApp().run()