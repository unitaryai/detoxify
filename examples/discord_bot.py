"""
Discord Moderation Bot using Detoxify and Discord.py
"""

import discord
from detoxify import Detoxify

client = discord.Client()

model = Detoxify('multilingual')


@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))


@client.event
async def on_message(message):
    if message.author == client.user:
        return
    prediction = model.predict(message.content)
    total_score = 0
    print(3 * '---')
    print(f"message: {message.content}")
    for key in prediction:
        print(f"{key}: {prediction[key]}")
        total_score += prediction[key]
    print(f"total_score: {total_score}")
    print(f"nsfw_channel: {message.channel.is_nsfw()}")
    if total_score > 2 and not message.channel.is_nsfw():
        await message.delete()
        await message.channel.send("Stop being so toxic! 😠")


client.run('your-token-here')
