import discord
import time
from discord.ext import commands

bot = commands.Bot()

@bot.slash_command(name="first_slash") #Add the guild ids in which the slash command will appear. If it should be in all, remove the argument, but note that it will take some time (up to an hour) to register the command if it's for all guilds.
async def first_slash(ctx): 
    await ctx.send("You executed the slash command!")

bot.run('APIKEY')