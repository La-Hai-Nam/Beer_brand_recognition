import _tkinter
import tkinter.filedialog
import tkinter.messagebox
import os
import tkinter
import tkinter.messagebox
from tkinter import filedialog
import customtkinter
from PIL import Image
import tensorflow as tf
import numpy as np
import webbrowser
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import cv2
from PIL import Image, ImageTk
from customtkinter import CTkImage

customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.percentage = 50.0
        self.title("Beer detection")
        self.geometry(f"{1100}x{500}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Beer OMat",
                                                 font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, command=self.directory, text="Select Image")
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
        self.threshold_label = customtkinter.CTkLabel(self.sidebar_frame, text="Threshold: 50%", anchor="w")
        self.threshold_label.grid(row=3, column=0, padx=20, pady=(10, 0))
        self.slider = customtkinter.CTkSlider(self.sidebar_frame, from_=0, to=100, command=self.slider_event, width= 150)
        self.slider.grid(row=4, column=0, padx=20, pady=(10, 10))
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame,
                                                                       values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame,
                                                               values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))
        self.exit_label = customtkinter.CTkLabel(self.sidebar_frame, text="Close Application:", anchor="w")
        self.exit_label.grid(row=9, column=0, padx=20, pady=(10, 0))
        self.exit_optionemenu = customtkinter.CTkButton(self.sidebar_frame,
                                                        command=lambda: self.destroy(),
                                                        text="Exit")
        self.exit_optionemenu.grid(row=10, column=0, padx=20, pady=(10, 20))
        # image
        self.slider_progressbar_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        self.slider_progressbar_frame.grid(row=1, column=1, columnspan=1, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.button_frames = customtkinter.CTkFrame(self, fg_color="transparent")
        self.button_frames.grid(row=1, column=2, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.result_label = customtkinter.CTkLabel(self.button_frames, text="")
        self.result_label.grid(row=0, column=0, padx=(20, 20), pady=(20, 0), sticky="nsew", rowspan=2)
        self.home_frame_large_image_label = customtkinter.CTkLabel(self,
                                                                   text="Please select an image \n\n\n\n\n\n\n\n\n ",
                                                                   fg_color="white", width=150, height=150,
                                                                   font=("Arial", 15), text_color="black")
        self.home_frame_large_image_label.grid(row=0, column=1, padx=(55, 0), pady=10)
        self.placeholder_label = customtkinter.CTkLabel(self, width=300, text="")
        self.placeholder_label.grid(row=0, column=2, padx=20, pady=10)
        self.offerButton = customtkinter.CTkButton(self.button_frames, fg_color="transparent", border_width=2,
                                                   text_color=("gray10", "#DCE4EE"),
                                                   text="Look for yourself",
                                                   state="disabled",
                                                   command=self.offer,
                                                   width=350,
                                                   height=50)
        self.offerButton.grid(row=2, column=0, padx=(20, 20), pady=(20, 0), sticky="nsew")
        self.tasteButton = customtkinter.CTkButton(self.button_frames, fg_color="transparent", border_width=2,
                                                   text_color=("gray10", "#DCE4EE"),
                                                   text="More Information about the beer",
                                                   state="disabled",
                                                   command=self.beerTaste,
                                                   width=350,
                                                   height=50)
        self.tasteButton.grid(row=3, column=0, padx=(20, 20), pady=(20, 0), sticky="nsew")
        self.screenshotButton = customtkinter.CTkButton(self.button_frames, fg_color="transparent", border_width=2,
                                                        text_color=("gray10", "#DCE4EE"),
                                                        text="Take camera snapshot",
                                                        state="normal",
                                                        command=self.snapshot,
                                                        width=350,
                                                        height=50)
        self.screenshotButton.grid(row=4, column=0, padx=(20, 20), pady=(20, 0), sticky="nsew")
        # set default values
        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("100%")
        self.percentageslabel1 = customtkinter.CTkLabel(self.slider_progressbar_frame, text="")
        self.percentageslabel1.grid(row=1, column=2, padx=(20, 10), pady=(1, 1), sticky="ew")
        self.percentageslabel2 = customtkinter.CTkLabel(self.slider_progressbar_frame, text="")
        self.percentageslabel2.grid(row=2, column=2, padx=(20, 10), pady=(1, 1), sticky="ew")
        self.percentageslabel3 = customtkinter.CTkLabel(self.slider_progressbar_frame, text="")
        self.percentageslabel3.grid(row=3, column=2, padx=(20, 10), pady=(1, 1), sticky="ew")
        self.percentageslabel4 = customtkinter.CTkLabel(self.slider_progressbar_frame, text="")
        self.percentageslabel4.grid(row=4, column=2, padx=(20, 10), pady=(1, 1), sticky="ew")
        self.percentageslabel5 = customtkinter.CTkLabel(self.slider_progressbar_frame, text="")
        self.percentageslabel5.grid(row=5, column=2, padx=(20, 10), pady=(1, 1), sticky="ew")
        self.percentageslabel6 = customtkinter.CTkLabel(self.slider_progressbar_frame, text="")
        self.percentageslabel6.grid(row=6, column=2, padx=(20, 10), pady=(1, 1), sticky="ew")
        self.percentageslabel7 = customtkinter.CTkLabel(self.slider_progressbar_frame, text="")
        self.percentageslabel7.grid(row=7, column=2, padx=(20, 10), pady=(1, 1), sticky="ew")
        self.percentageslabel8 = customtkinter.CTkLabel(self.slider_progressbar_frame, text="")
        self.percentageslabel8.grid(row=8, column=2, padx=(20, 10), pady=(1, 1), sticky="ew")

        self.pb1_name = customtkinter.CTkLabel(self.slider_progressbar_frame,
                                               text="Becks")
        self.pb1_name.grid(row=1, column=0, padx=(20, 10), pady=(1, 1), sticky="ew")

        self.progressbar_1 = customtkinter.CTkProgressBar(self.slider_progressbar_frame,
                                                          progress_color="green")
        self.progressbar_1.set(0.0)
        self.progressbar_1.grid(row=1, column=1, padx=(20, 10), pady=(10, 10), sticky="ew")
        self.pb2_name = customtkinter.CTkLabel(self.slider_progressbar_frame,
                                               text="Berliner Kindl")
        self.pb2_name.grid(row=2, column=0, padx=(20, 10), pady=(1, 1), sticky="ew")

        self.progressbar_2 = customtkinter.CTkProgressBar(self.slider_progressbar_frame,
                                                          progress_color="green")
        self.progressbar_2.set(0.0)
        self.progressbar_2.grid(row=2, column=1, padx=(20, 10), pady=(1, 1), sticky="ew")
        self.pb3_name = customtkinter.CTkLabel(self.slider_progressbar_frame,
                                               text="Corona")
        self.pb3_name.grid(row=3, column=0, padx=(20, 10), pady=(1, 1), sticky="ew")

        self.progressbar_3 = customtkinter.CTkProgressBar(self.slider_progressbar_frame,
                                                          progress_color="green")
        self.progressbar_3.set(0.0)
        self.progressbar_3.grid(row=3, column=1, padx=(20, 10), pady=(10, 10), sticky="ew")
        self.pb4_name = customtkinter.CTkLabel(self.slider_progressbar_frame,
                                               text="Krombacher")
        self.pb4_name.grid(row=4, column=0, padx=(20, 10), pady=(1, 1), sticky="ew")
        self.progressbar_4 = customtkinter.CTkProgressBar(self.slider_progressbar_frame,
                                                          progress_color="green")
        self.progressbar_4.set(0.0)
        self.progressbar_4.grid(row=4, column=1, padx=(20, 10), pady=(10, 10), sticky="ew")
        self.pb5_name = customtkinter.CTkLabel(self.slider_progressbar_frame,
                                               text="Krombacher alkoholfrei")
        self.pb5_name.grid(row=5, column=0, padx=(20, 10), pady=(1, 1), sticky="ew")
        self.progressbar_5 = customtkinter.CTkProgressBar(self.slider_progressbar_frame,
                                                          progress_color="green")
        self.progressbar_5.set(0.0)
        self.progressbar_5.grid(row=5, column=1, padx=(20, 10), pady=(10, 10), sticky="ew")
        self.pb6_name = customtkinter.CTkLabel(self.slider_progressbar_frame,
                                               text="Mönchshof")
        self.pb6_name.grid(row=6, column=0, padx=(20, 10), pady=(1, 1), sticky="ew")
        self.progressbar_6 = customtkinter.CTkProgressBar(self.slider_progressbar_frame,
                                                          progress_color="green")
        self.progressbar_6.set(0.0)
        self.progressbar_6.grid(row=6, column=1, padx=(20, 10), pady=(10, 10), sticky="ew")
        self.pb7_name = customtkinter.CTkLabel(self.slider_progressbar_frame,
                                               text="Sternburg")
        self.pb7_name.grid(row=7, column=0, padx=(20, 10), pady=(1, 1), sticky="ew")
        self.progressbar_7 = customtkinter.CTkProgressBar(self.slider_progressbar_frame,
                                                          progress_color="green")
        self.progressbar_7.set(0.0)
        self.progressbar_7.grid(row=7, column=1, padx=(20, 10), pady=(10, 10), sticky="ew")
        self.pb8_name = customtkinter.CTkLabel(self.slider_progressbar_frame,
                                               text="Warsteiner")
        self.pb8_name.grid(row=8, column=0, padx=(20, 10), pady=(1, 1), sticky="ew")
        self.progressbar_8 = customtkinter.CTkProgressBar(self.slider_progressbar_frame,
                                                          progress_color="green")
        self.progressbar_8.set(0.0)
        self.progressbar_8.grid(row=8, column=1, padx=(20, 10), pady=(10, 10), sticky="ew")

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def directory(self):

        try:
            self.filepath = filedialog.askopenfilename()
            im = Image.open(self.filepath)
            f, e = os.path.splitext(self.filepath)
            imResize = im.resize((150, 150), Image.Resampling.LANCZOS)
            imResize.save(f + ".png", 'PNG', quality=90)

            if self.filepath:
                # Display the image in a label widget
                self.image = tkinter.PhotoImage(file=self.filepath)
                self.home_frame_large_image_label.configure(image=self.image, text="")
                self.test_model(imResize)
        except _tkinter.TclError:
            window = customtkinter.CTkToplevel(self)
            window.geometry("400x200")
            window.title("Warning!")

            # create label on CTkToplevel window
            label = customtkinter.CTkLabel(window, text="You have selected a not supported image format e.g. JPEG\n"
                                                        "Only PNG is supported. We have however\ncreated a png copy of your selected image in the same directory,\n"
                                                        "you just have to select the png copy again :)")
            label.pack(side="top", fill="both", expand=True, padx=15, pady=20)
            button = customtkinter.CTkButton(window, text="Okay, select image",
                                             command=lambda: [f() for f in [window.destroy, self.directory]])
            button.pack(side="bottom", fill="both", expand=True, padx=15, pady=20)

    def test_model(self, userInput):
        image = userInput
        # Load the TensorFlow Lite model
        interpreter = tf.lite.Interpreter(model_path="beerDetectionModel.tflite")
        interpreter.allocate_tensors()

        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Read the sample JPG image and pre-process it for TensorFlow Lite

        image_data = np.array(image, dtype=np.float32)

        # Set the input tensor to the image data
        interpreter.set_tensor(input_details[0]['index'], [image_data])

        # Run the model
        interpreter.invoke()

        # Get the output from the model
        self.class_names = ["Becks", "Berliner-Kindl", "Corona", "Krombacher", "Krombacher-Alkoholfrei", "Moenchshof",
                            "Sternburg", "Warsteiner"]
        output_data = interpreter.get_tensor(output_details[0]['index'])
        self.score = tf.nn.softmax(output_data)
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(self.class_names[np.argmax(self.score)], 100 * np.max(self.score))
        )
        self.progressbar_1.set(np.max(self.score[0][0]))
        self.percentageslabel1.configure(text=str('{:.2f}'.format(100 * np.max(self.score[0][0]))) + " %")
        self.progressbar_2.set(np.max(self.score[0][1]))
        self.percentageslabel2.configure(text=str('{:.2f}'.format(100 * np.max(self.score[0][1]))) + " %")
        self.progressbar_3.set(np.max(self.score[0][2]))
        self.percentageslabel3.configure(text=str('{:.2f}'.format(100 * np.max(self.score[0][2]))) + " %")
        self.progressbar_4.set(np.max(self.score[0][3]))
        self.percentageslabel4.configure(text=str('{:.2f}'.format(100 * np.max(self.score[0][3]))) + " %")
        self.progressbar_5.set(np.max(self.score[0][4]))
        self.percentageslabel5.configure(text=str('{:.2f}'.format(100 * np.max(self.score[0][4]))) + " %")
        self.progressbar_6.set(np.max(self.score[0][5]))
        self.percentageslabel6.configure(text=str('{:.2f}'.format(100 * np.max(self.score[0][5]))) + " %")
        self.progressbar_7.set(np.max(self.score[0][6]))
        self.percentageslabel7.configure(text=str('{:.2f}'.format(100 * np.max(self.score[0][6]))) + " %")
        self.progressbar_8.set(np.max(self.score[0][7]))
        self.percentageslabel8.configure(text=str('{:.2f}'.format(100 * np.max(self.score[0][7]))) + " %")

        if float(self.percentage) <= float(100 * np.max(self.score)):
            self.placeholder_label.configure(
                text="This image most likely belongs\n to {} with\n a {:.2f} percent confidence."
                .format(self.class_names[np.argmax(self.score)], 100 * np.max(self.score)), font=("Arial", 20))
            self.offerButton.configure(state="normal")
            self.tasteButton.configure(state="normal")
            self.url = "https://www.kaufda.de/Angebote/"
            self.url += str(self.class_names[np.argmax(self.score)])
            self.crawl()

        else:
            self.placeholder_label.configure(
                text="There is no definite result.\n Every value is below\nthe configured threshold.", font=("Arial", 20))
            self.offerButton.configure(state="disabled")
            self.tasteButton.configure(state="disabled")


    def offer(self):
        webbrowser.open(self.url, new=0, autoraise=True)

    def crawl(self):
        try:
            # Make a request to the website
            response = requests.get(self.url)
            soup = BeautifulSoup(response.content, 'html.parser')
            # Find a specific element on the page
            name = soup.find('div', attrs={'class': 'offer-retailer ellipsis'})
            price = soup.find('p', attrs={'class': 'offer-price ellipsis'})
            # Extract the desired information from the element
            # nameInfo = name.text
            # priceInfo = price.text
            result_str = "There is currently an offer at:\n" + name.text + " for " + price.text
            self.result_label.configure(text=result_str, font=("Arial", 20))
        except AttributeError:
            result_str = "There is currently no offer"
            self.result_label.configure(text=result_str, font=("Arial", 20))

    def beerTaste(self):
        tree = ET.parse('data.xml')

        # Get the root element
        root = tree.getroot()
        string = "beer" + str(np.argmax(self.score))
        # Find the element1 element
        element1 = root.find(string)

        # Find all subelement elements within element1
        name = element1.findall('.//Name')
        sort = element1.findall('.//Sort')
        alc = element1.findall('.//Alcohol')
        ingredients = element1.findall('.//Ingredients')
        color = element1.findall('.//Color')
        foam = element1.findall('.//Foam')
        smell = element1.findall('.//Smell')
        taste = element1.findall('.//Taste')

        # Print the text of all subelement elements within element1
        window = customtkinter.CTkToplevel(self)
        window.geometry("400x600")
        window.title("Additional Information")

        # create label on CTkToplevel window
        label = customtkinter.CTkLabel(window, text=name[0].text, font=("Arial", 20))
        label.pack(side="top", fill="both", expand=False, padx=15, pady=20)
        label2 = customtkinter.CTkLabel(window, text="Alcohol:\n" + alc[0].text, font=("Arial", 15))
        label2.pack(side="top", fill="both", expand=False, padx=15, pady=10)
        label9 = customtkinter.CTkLabel(window, text="Type:\n" + sort[0].text, font=("Arial", 15))
        label9.pack(side="top", fill="both", expand=False, padx=15, pady=10)
        label3 = customtkinter.CTkLabel(window, text="Ingredients:\n" + ingredients[0].text, font=("Arial", 15))
        label3.pack(side="top", fill="both", expand=False, padx=15, pady=10)
        label4 = customtkinter.CTkLabel(window, text="Color:\n" + color[0].text, font=("Arial", 15))
        label4.pack(side="top", fill="both", expand=False, padx=15, pady=10)
        label5 = customtkinter.CTkLabel(window, text="Foam:\n" + foam[0].text, font=("Arial", 15))
        label5.pack(side="top", fill="both", expand=False, padx=15, pady=10)
        label6 = customtkinter.CTkLabel(window, text="Smell:\n" + smell[0].text, font=("Arial", 15))
        label6.pack(side="top", fill="both", expand=False, padx=15, pady=10)
        label7 = customtkinter.CTkLabel(window, text="Taste:\n" + taste[0].text, font=("Arial", 15))
        label7.pack(side="top", fill="both", expand=False, padx=15, pady=10)
        label8 = customtkinter.CTkLabel(window, text="Source: http://www.1000getraenke.de/biertest/",
                                        font=("Arial", 10), text_color="#3366CC")
        label8.pack(side="top", fill="both", expand=False, padx=15, pady=10)
        label8.bind("<Button-1>", lambda e: self.callback("http://www.1000getraenke.de/biertest/"))
        button = customtkinter.CTkButton(window, text="Close window", command=window.destroy)
        button.pack(side="bottom", fill="both", expand=False, padx=15, pady=20)

    def callback(self, url):
        webbrowser.open(url, new=0, autoraise=True)

    def snapshot(self):
        try:
            self.cap = cv2.VideoCapture(0)
            self.window = customtkinter.CTkToplevel(self)
            self.window.title("Webcam Live Feed")
            self.label = customtkinter.CTkLabel(self.window, text="")
            self.label.pack()
            self.button = customtkinter.CTkButton(self.window, text="Take Screenshot", command=lambda: [f() for f in [
                self.take_screenshot, self.window.destroy()]],  # still get error here
                                                  fg_color="transparent", border_width=2,
                                                  text_color=("gray10", "#DCE4EE"),
                                                  width=350,
                                                  height=50)
            self.button.pack(padx=20, pady=20)
            self.update_frame()
        except cv2.error:  # fixed bug with opening snapshot window twice in a row
            self.window.destroy()
            self.snapshot()

    def update_frame(self):
        # Read the frame from the webcam
        ret, frame = self.cap.read()

        # Convert the frame to a PhotoImage object
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        ctk_image = customtkinter.CTkImage(light_image=frame, size=(600, 450))
        # Update the label with the new frame
        self.label.configure(image=ctk_image)
        self.label.image = ctk_image

        # Schedule the update_frame function to be called after 30 milliseconds
        self.window.after(30, self.update_frame)

    # Function to take a screenshot
    def take_screenshot(self):
        # Read the frame from the webcam
        ret, frame = self.cap.read()

        # Save the frame to a file
        cv2.imwrite("screenshot.png", frame)

        im = Image.open("screenshot.png")
        imResize = im.resize((150, 150), Image.Resampling.LANCZOS)
        imResize.save("screenshot.png", 'PNG', quality=90)

        # Display the image in a label widget
        image = tkinter.PhotoImage(file="screenshot.png")
        self.home_frame_large_image_label.configure(image=image, text="")
        self.test_model(imResize)

    def slider_event(self, value):
        self.percentage = "{:.2f}".format(value)
        threshold_str = "Threshold: " + self.percentage + "%"
        self.threshold_label.configure(text=threshold_str)


if __name__ == "__main__":
    app = App()
    app.mainloop()
