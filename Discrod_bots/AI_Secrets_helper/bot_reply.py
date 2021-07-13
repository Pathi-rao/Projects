import discord

#client = discord.Client()

class MyClient(discord.Client):

  async def on_message(self, message):
      # we do not want the bot to reply to itself
      if message.author.id == self.user.id:
          return

      # *********************************************************** #
      # Basic interaction
  
      elif message.content.lower().startswith('!hello'):
        await message.reply('Hi there!', mention_author=True)

      elif message.content.lower().startswith('!about'):
        await message.channel.send("Hello, I am a discord bot. I am here to make your life a bit easy. If you have any other cool features in mind, please feel free to reach the admin. You can always ask me for help using `!help` command.")

      elif message.content.lower().startswith('!help'):
        await message.channel.send(" What can I help you with?. \n Hint: You can type `!commands` to get the list of commands that I currently support.")

      # *********************************************************** #
      # list of commands

      elif message.content.lower().startswith('!commands'):
        await message.channel.send("These are the list of commands that you can use. \n \n \
        To get the calendly links of Tutors:\t `!*name*-1o1` \n Eg: `!Elon-1o1` \n \n \
        To get the github links of peers: \t `!git-*name*` \n Eg: `!git-Elon` \n  To know about the admin: \t`!admin` ")
      
      # *********************************************************** #
      # Calendly links

      elif message.content.lower().startswith('!antonio-1o1'):
        await message.channel.send("https://calendly.com/antonio-marsella")

      elif message.content.lower().startswith('!jon-1o1'):
        await message.channel.send("https://calendly.com/jonperezetxebarria")

      elif message.content.lower().startswith('!elon-1o1'):
        await message.channel.send("Nice try but Elon Musk is currently busy. Try contacting the tutors instead.")

      # *********************************************************** #
      # Github profiles

      elif message.content.lower().startswith('!git-pathi'):
        await message.channel.send("https://github.com/Pathi-rao")

      elif message.content.lower().startswith('!git-alessio'):
        await message.channel.send("https://github.com/alessiorecchia")

      elif message.content.lower().startswith('!git-deniz'):
        await message.channel.send("https://github.com/Deniz-shelby")

      elif message.content.lower().startswith('!git-dilan'):
        await message.channel.send("https://github.com/UdawalaHewageDilan")

      elif message.content.lower().startswith('!git-fabio'):
        await message.channel.send("https://github.com/fistadev")

      elif message.content.lower().startswith('!git-sven'):
        await message.channel.send("https://github.com/Sven-Skyth-Henriksen")

      elif message.content.lower().startswith('!git-joshua'):
        await message.channel.send("https://github.com/Josh-Batt")

      elif message.content.lower().startswith('!git-gabriel'):
        await message.channel.send("https://github.com/Calypso25")
      
      elif message.content.lower().startswith('!git-umut'):
        await message.channel.send("https://github.com/aktumut")

      elif message.content.lower().startswith('!git-elon'):
        await message.channel.send("Nice try but Elon is not your peer.Try accessing your other peers.")

      # *********************************************************** #
      # About the admin

      elif message.content.lower().startswith('!admin'):
        await message.channel.send("Hi, My name is Pathi. I developed this bot because I am too lazy to remeber things and also to keep everthing at one place. If you want to contribute to this project or have any suggestions, please feel free to reach out to me.")


      # *********************************************************** #
      # if the command doesn't exist

      elif message.content.lower().startswith('!'):
        await message.channel.send("Sorry, I don't have that command yet. Feel free to open a request to the admin to add this command.")

