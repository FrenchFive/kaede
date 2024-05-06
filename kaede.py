import nextcord as discord
from nextcord.ext import commands

from datetime import datetime

import random

import os
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionPipeline
from diffusers.utils import make_image_grid
import torch

import json

def filecretor(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename+'-'+str(counter)+extension
        counter+=1
    return path

def stablediff(message, arw, arh, seeding, fname, step, cfg, style):
    prompt = message
    neg_prompt = 'blurry, ugly, bad, low quality, lowres'
    w=arw
    h=arh
    steps = step
    denoising = 0.8
    cfgscale = cfg
    seed = seeding

    generator = torch.Generator(device="cuda").manual_seed(seed)

    global base
    global refiner
    global SAVE_PATH
    if style == "normal":
        image = base(
            prompt=prompt,
            negative_prompt=neg_prompt,
            width=w,
            height=h,
            guidance_scale=cfgscale,
            generator = generator,
            num_inference_steps=steps,
            denoising_end=denoising,
            output_type="latent",
        ).images

        image = refiner(
            prompt=prompt,
            negative_prompt=neg_prompt,
            guidance_scale=cfgscale,
            generator = generator,
            num_inference_steps=steps,
            denoising_start=denoising,
            image=image,
        ).images[0]

    elif style == "anime":
        anime = StableDiffusionPipeline.from_single_file(
            ANIME_MODEL_PATH,
            torch_dtype=torch.float16, 
            variant="fp16",
            safety_checker = None,
            requires_safety_checker = False,
            use_safetensors=True
        ).to("cuda")
        anime.safety_checker = None
        anime.requires_safety_checker = False
        image = anime(
            prompt=prompt,
            negative_prompt=neg_prompt,
            width=w,
            height=h,
            guidance_scale=cfgscale,
            generator = generator,
            num_inference_steps=steps,
        ).images[0]
        del anime
    elif style == "realistic":
        realistic = StableDiffusionPipeline.from_single_file(
            REALISTIC_MODEL_PATH,
            torch_dtype=torch.float16, 
            variant="fp16",
            safety_checker = None,
            requires_safety_checker = False,
            use_safetensors=True
        ).to("cuda")
        realistic.safety_checker = None
        realistic.requires_safety_checker = False
        image = realistic(
            prompt=prompt,
            negative_prompt=neg_prompt,
            width=w,
            height=h,
            guidance_scale=cfgscale,
            generator = generator,
            num_inference_steps=steps,
        ).images[0]
        del realistic
    elif style == "flat":
        flat = StableDiffusionPipeline.from_single_file(
            FLAT_MODEL_PATH,
            torch_dtype=torch.float16, 
            variant="fp16",
            safety_checker = None,
            requires_safety_checker = False,
            use_safetensors=True
        ).to("cuda")
        flat.safety_checker = None
        flat.requires_safety_checker = False
        image = flat(
            prompt=prompt,
            negative_prompt=neg_prompt,
            width=w,
            height=h,
            guidance_scale=cfgscale,
            generator = generator,
            num_inference_steps=steps,
        ).images[0]
        del flat

    image_path = filecretor(os.path.join(SAVE_PATH,fname+'.png'))

    image.save(image_path)
    return image_path, image

def jswrite(cmd, author, message, seed, message_id, imgpath, url):
    global JSON_FILE
    file_path = JSON_FILE

    # Read existing JSON data
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = []

    # Create a new message dictionary
    new_message = {
        "Author": author,
        "Message": message,
        "Command": cmd,
        "Seed": seed,
        "Image Path": imgpath,
        "Message ID": message_id,
        "URL": url,
        "Time": str(datetime.now().strftime('%d/%m/%Y %H:%M:%S'))
    }

    # Append the new message to the existing data
    data.append(new_message)

    # Write the updated data back to the JSON file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=7)

def jsread(messid):
    global JSON_FILE
    file_path = JSON_FILE
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            for message in data:
                if message.get("Message ID") == messid:
                    return message
    except FileNotFoundError:
        return None

def dream(args, author):
    cmd = ' '.join(args)
    if '--style' in args:
        pos = args.index('--style')
        liofstyles = ["anime","realistic","flat","normal"]
        style = args.pop(pos+1)
        if style not in liofstyles:
            style = "normal"
        del args[pos]
    else:
        style = "normal"
    
    if '--ar' in args:
        liaar = ['1024:1024','1152:896','1216:832','1344:768','1536:640','640:1536','768:1344','832:1216','896:1152']
        liaarcal = [1, 1.28, 1.4, 1.75, 2.4, 0.42, 0.57, 0.68, 0.77]
        pos = args.index('--ar')
        ratio = args.pop(pos+1)
        del args[pos]
        ratio = ratio.split(':')
        ratio[0] = int(ratio[0])
        ratio[1] = int(ratio[1])
        cal = ratio[0]/ratio[1]
        closestval = min(liaarcal, key=lambda liaarcal :abs(liaarcal-cal))
        closest = liaarcal.index(closestval)
        ar = liaar[closest].split(':')
        arw = int(ar[0])
        arh = int(ar[1])
    else :
        arw = 1024
        arh = 1024

    if '--steps' in args:
        pos = args.index('--steps')
        steps = int(args.pop(pos+1))
        del args[pos]
    else:
        steps = 50

    if '--cfg' in args:
        pos = args.index('--cfg')
        cfg = int(args.pop(pos+1))
        del args[pos]
    else:
        cfg = 5
    
    if '--seed' in args:
        pos = args.index('--seed')
        seeding = int(args.pop(pos+1))
        del args[pos]
    else :
        seeding = random.randint(0,10000000)

    if '--batch' in args:
        loop = 4
        steps = 15
        del args[args.index('--batch')]
    else:
        loop = 1

    message = ' '.join(args)
    if message == '':
        message = 'Something random'
    
    now = datetime.now()
    formatted = now.strftime("%H:%M:%S")
    print(f'{formatted} // PAINTING : {message[:20]} - by {author}')
    
    liimages=[]
    liseed=[]
    for i in range (loop):
        if loop>1:
            seeding = random.randint(0,10000000)
        
        #FILENAME CREATION
        fname = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S_'))
        fname+= message[:20]+'_'+ str(seeding) +'_'
        fname+= str(author)

        imgpath, img = stablediff(message, arw, arh, seeding, fname, steps, cfg, style)
        liimages.append(img)
        liseed.append(str(seeding))
        
    if loop>1:
        seeding = ' / '.join(liseed)
        grid = make_image_grid(liimages, 2, 2)
        imgpath = filecretor(os.path.join(SAVE_PATH,fname+'_GRID'+'.png'))
        grid.save(imgpath)
    
    return imgpath, cmd, message, seeding

SDXL_MODEL_PATH = "G:/Chan/Documents/zKaede/KAEDE-V2/stable-diffusion-xl-base-1.0"
SDXL_REFINER_MODEL_PATH = "G:/Chan/Documents/zKaede/KAEDE-V2/stable-diffusion-xl-refiner-1.0"
ANIME_MODEL_PATH = "G:/Chan/Documents/zKaede/KAEDE-V2/models/arthemy.safetensors"
REALISTIC_MODEL_PATH = "G:/Chan/Documents/zKaede/KAEDE-V2/models/realisticVision.safetensors"
FLAT_MODEL_PATH = "G:/Chan/Documents/zKaede/KAEDE-V2/models/flat2DAnimerge.safetensors"
SAVE_PATH = "G:/Chan/Documents/zKaede/KAEDE-V2/OUTPUT"
JSON_FILE = "G:/Chan/Documents/zKaede/KAEDE-V2/data.json"

base = DiffusionPipeline.from_pretrained(
    SDXL_MODEL_PATH, 
    torch_dtype=torch.float16, 
    variant="fp16", 
    use_safetensors=True
).to("cuda")

refiner = DiffusionPipeline.from_pretrained(
    SDXL_REFINER_MODEL_PATH,
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
).to("cuda")


intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='.', intents=intents)
bot.remove_command('help')

@bot.event
async def on_ready():
    print("KAEDE IS ONLINE")

@bot.event
async def on_raw_reaction_add(payload):
    user = payload.member
    channel = bot.get_channel(payload.channel_id)
    if user.name == bot.user.name:
        return
    if str(payload.emoji) == '\U0001F194':
        info = jsread(payload.message_id)
        if info != None:
            embed= discord.Embed(
                title=info['Message'],
                description = '.imagine ' + str(info['Command']),
                color=discord.Color.blurple()
            )
            embed.set_thumbnail(url=info['URL'])
            embed.add_field(
                name="Author",
                value=info['Author'],
                inline=True
            )
            embed.add_field(
                name="Seed",
                value=info['Seed'],
                inline=True
            )
            embed.add_field(
                name="Time",
                value=info['Time'],
                inline=True
            )
            embed.add_field(
                name="Message ID",
                value=info['Message ID'],
                inline=True
            )
            embed.set_footer(text="Image generated by Kaede.")
            await channel.send(embed=embed)
    if str(payload.emoji) == '\U0001F504':
        info = jsread(payload.message_id)
        if info != None:
            cmdjs = info['Command']
            args = list(cmdjs.split(" "))
            author = user.name
            imgpath, cmd, message, seeding = dream(args, author)
            messent = await channel.send(file=discord.File(imgpath))

            message_id = messent.id
            url = messent.attachments[0].url
            jswrite(cmd, author, message, seeding, message_id, imgpath, url)

            emojireact = ['\U0001F504','\U0001F194']
            for i in range (len(emojireact)):
                await messent.add_reaction(emojireact[i])

@bot.command()
async def help(ctx):
    embed= discord.Embed(
        title="HELP - KAEDE",
        description="Kaede is an AI Generative Art Bot.",
        color=discord.Color.purple()
    )
    #embed.set_thumbnail(url="./out/txt2img_2408908424.png")
    embed.add_field(
        name=".imagine",
        value='Allows you to create beautiful artwork from text [.imagine a woman]',
        inline=False
    )
    embed.add_field(
        name="--style",
        value='Let you change the Model (by default normal == SDXL) [.imagine a woman --style anime] {normal, anime, realistic, flat}',
        inline=True
    )
    embed.add_field(
        name="--ar",
        value='Let you change the Aspect Ratio [.imagine a woman --ar 9:16]',
        inline=True
    )
    embed.add_field(
        name="--seed",
        value='Let you change the Seed [.imagine a woman --seed 2910]',
        inline=True
    )
    embed.add_field(
        name="--batch",
        value='Let you generate 4 images instead of 1 (lower quality) [.imagine a woman --batch]',
        inline=True
    )
    embed.add_field(
        name="--steps",
        value='Let you control how many time Kaede will draw over the image (default - 50) [.imagine a woman --steps 20]',
        inline=True
    )
    embed.add_field(
        name="--cfg",
        value='Let you control how much Kaede will try to get close to the prompt (default - 5) [.imagine a woman --cfg 15]',
        inline=True
    )
    embed.set_footer(text="Kaede is a coding experience made by Five (alias Chan)")
    await ctx.send(embed=embed)

@bot.command()
async def imagine(ctx, *args):
    args = list(args)

    liemoji = ['\U0001F498','\U0001F5A4','\U0001F44C','\U0001FAF6','\U0001F440','\U0001FA84']
    emoji = liemoji[random.randint(0,len(liemoji)-1)]
    await ctx.message.add_reaction(emoji)

    author = ctx.author.name
    imgpath, cmd, message, seeding = dream(args, author)
    messent = await ctx.send(file=discord.File(imgpath))

    message_id = messent.id
    url = messent.attachments[0].url
    jswrite(cmd, author, message, seeding, message_id, imgpath, url)

    emojireact = ['\U0001F504','\U0001F194']
    for i in range (len(emojireact)):
        await messent.add_reaction(emojireact[i])

@bot.command()
async def valagent(ctx, *args):
    args = list(args)
    agents = ['Brimstone','Phoenix','Sage','Sova','Viper','Cypher','Reyna','Killjoy','Breach','Omen','Jett','Raze','Skye','Yoru','Astra','Kay/o','Chamber','Neon','Fade','Harbor','Gekko','Deadlock','Iso']
    for i in args:
            index = random.randint(0,len(agents)-1)
            sending = str(i) + " :: " + agents[index].upper()
            del(agents[index])
            await ctx.send(sending)

@bot.command()
async def valmap(ctx):
    maps = ['Sunset','Lotus','Pearl','Fracture','Breeze','Icebox','Bind','Haven','Split','Ascent']
    index = random.randint(0,len(maps)-1)
    map = str(maps[index])
    await ctx.send("The Map shall be :: " + map.upper())

@bot.command()
async def valweapon(ctx, *args):
    args = list(args)
    weapons = ['Classic','Shorty','Frenzy','Ghost','Sheriff','Stinger','Spectre','Bucky','Judge','Bulldog','Guardian','Phantom','Vandal','Marshal','Operator','Ares','Odin','Knife']
    if len(args) <= 0:
        index = random.randint(0,len(weapons)-1)
        weapon = str(weapons[index])
        await ctx.send("The weapon shall be :: " + weapon.upper())
    else:
        for i in args:
            index = random.randint(0,len(weapons)-1)
            weapon = str(weapons[index])
            sending = f"{i}'s weapon :: {weapon.upper()}"
            await ctx.send(sending)

@bot.command()
async def daily(ctx, *args):
    probepic = 0.001
    probrare = 0.01
    prob = random.random()
    
    if prob <= probepic:
        card = 'Epic'
    elif prob <= probrare:
        card = 'Rare'
    else:
        card = 'Common'

    await ctx.send(f"YOU WON A ... {card.upper()} CARD")

bot.run('APIKEY')