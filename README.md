# Llama 3.2 Multimodal
Meta has recently launched it's new set of large language models and are calling it Llama 3.2. The two largest models of the Llama 3.2 collection, 11B and 90B, support image reasoning use cases, such as document-level understanding including charts and graphs, captioning of images, and visual grounding tasks such as directionally pinpointing objects in images based on natural language descriptions. 

For example, a person could ask a question about which month in the previous year their small business had the best sales, and Llama 3.2 can then reason based on an available graph and quickly provide the answer. In another example, the model could reason with a map and help answer questions such as when a hike might become steeper or the distance of a particular trail marked on the map. The 11B and 90B models can also bridge the gap between vision and language by extracting details from an image, understanding the scene, and then crafting a sentence or two that could be used as an image caption to help tell the story.

![Llama](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1a4fd651-e75f-47a6-8280-b66a78d25bbe_800x322.png)

In this repository, we are going to use the Llama 3.2 90b parameters model using fireworks.ai api to run inference on image inputs and then asking the model questions related to the image in the form of prompts. We will also use the [Instructor](https://pypi.org/project/instructor/) library to help us provide structured outputs from large language models (LLMs)

## Input (Financial Chart)
![Financial Chart](https://cdn.boldbi.com/wp/blogs/unlocking-financial-insights/area-chart-example.webp)
## Output 
```
chart_title='Revenue Growth Trend' 

key_findings='Revenue peaked in November at $927K.', 'There was a significant increase in revenue from July to November.', 'Revenue declined by 25% from November to December.'

insights='The chart shows a steady increase in revenue from July to November, with a significant spike to $927K in November. However, it declines by 25% in December.'
```

## Input Image
![Toy Story](https://mickeyblog.com/wp-content/uploads/2018/11/2018-11-05-20_41_02-Toy-Story-4_-Trailer-Story-Cast-Every-Update-You-Need-To-Know-720x340.png)

## Output Text
```
movie_name='Toy Story'

movie_rate='9.5'

movie_review="This is classic Pixar movie about toys that talk. It's hard not to love the classic character of Woody Cowboy, a Sheriff toy with a pull string that speaks in a twangy drawl. This film is like a daydream come true, what if toys were aloud talk when no human was around?"
```


## Input Image
![Idli Smabhar](https://www.awesomecuisine.com/wp-content/uploads/2023/03/Idli-sambhar-food.png)

## Output Text
```
recipe_name='Idli Sambar'

ingredients=['Idli batter', 'Split yellow lentils', 'Toor dal', 'Urad dal', 'Mustard seeds', 'Cumin seeds', 'Curry leaves', 'Onion', 'Tomato', 'Sambar powder', 'Tamarind paste', 'Jaggery', 'Salt', 'Coconut oil', 'Water']

instructions='Prepare the idli batter and let it ferment. Then, steam the idlis according to the package instructions. For the sambar, heat oil in a pan and add mustard seeds and cumin seeds. Saute the onions and tomatoes, then add the lentils and spices. Add tamarind paste and jaggery, and let it simmer. Finally, serve the idlis with the sambar.'
```

## Image captured through live feed
![Coca Cola Zero](https://github.com/AnshChoudhary/llama-3.2-multimodal/blob/main/screenshot.jpg)

## Output Text
Available cameras: [0, 1]
 
Enter the camera index to use: 1

Initializing camera... Please wait.

Camera feed opened. Press 'q' to capture an image.

Image captured successfully.

Enter your prompt for the image: what do you see in this image? what is the person holding?


The image appears to be a photograph of someone holding up a bottle with a red label in front of a wall with orange stripes and a dart board hanging. It is not possible to see the person holding the bottle but looks more like they might be taking a selfie with the bottle in front of them.

There are also wooden cupboards containing a mirror displayed on the left-hand side of the wall. In their right hand, the person is holding a Coca-Cola bottle marked “"Zero Sugar" with the Coca-Cola signature font written on a red label. Directly below it is a white label stating that the bottle is made from “100% recycled plastic”. Below that, the bottle is dark as the Coke is clearly contained within it. Above the bottle, there appears to be some water in the top of the bottle or perhaps condensation on the bottle. It is difficult to tell which, however, owing to the bad quality of the picture. Overall, the image displays a Coca-Cola item from front. Whether this was intended to be a selfie of a person with a bottle or a close-up of a Coca-Cola bottle with the person in the background is unclear owing to the quality of the picture.

# JARVIS

If you have watched any of the Iron Man movies, you must be familiar with the AI voice assistant JARVIS. Creating something like JARVIS has always been a dream of mine. With the new Llama 3.2 multimodal model, it is now possible to have an AI-powered voice assistant that is also able to see and comprehend pictures. Adding this feature to JARVIS would mean that you could ask JARVIS anything about whatever you are looking at just by asking it to run an inference on the frame taken from the camera live feed. 
