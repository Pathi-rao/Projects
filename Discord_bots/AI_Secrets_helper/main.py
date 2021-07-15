import discord
from bot_active import bot_status
from bot_reply import MyClient

#import env variables which contains our bot token
import os
my_secret = os.environ['token']


""" create instance of a client (this is the connection to discord) """

client = discord.Client()

#we can create as many callback events as we want
#event 1 --> when user is logged in
@client.event
async def on_ready():
  print('logged in as {0.user}' .format(client)) # the 0 will be replace with the username


#event 2 --> when bot receives a message
client = MyClient()


#to run the bot
bot_status()
client.run(my_secret) #token