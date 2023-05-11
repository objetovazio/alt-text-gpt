import logging
from telegram import Update
from telegram.ext import filters, MessageHandler, ApplicationBuilder, CommandHandler, ContextTypes
import os
from telegram import InlineQueryResultArticle, InputTextMessageContent
from telegram.ext import InlineQueryHandler

import replicate

BOT_TOKEN = os.getenv('BOT_TOKEN')
os.environ.get("REPLICATE_API_TOKEN")

logging.basicConfig(
    filename='./output.log',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user
    intro_message = 'Olá {}! Eu sou o {}.'.format(user['username'], context.bot.name)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=intro_message)
    description = "Eu sou capaz de gerar texto alternativo através de uma imagem. Me envie uma imagem em formato de arquivo, sem compressão para iniciar."
    await context.bot.send_message(chat_id=update.effective_chat.id, text=description)

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text=update.message.text)

async def caps(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text_caps = ' '.join(context.args).upper()
    await context.bot.send_message(chat_id=update.effective_chat.id, text=text_caps)

async def inline_caps(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.inline_query.query
    if not query:
        return
    results = []
    results.append(
        InlineQueryResultArticle(
            id=query.upper(),
            title='Caps',
            input_message_content=InputTextMessageContent(query.upper())
        )
    )
    await context.bot.answer_inline_query(update.inline_query.id, results)

async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Comando inexistente.")

async def image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file = update.message.photo[0].file_id
    obj = await context.bot.get_file(file)
    obj.download()
    update.message.reply_text("Image received")

async def downloader(update, context):
    file = await context.bot.get_file(update.message.document)
    await context.bot.send_message(chat_id=update.effective_chat.id, text='Baixando imagem...')
    await file.download_to_drive('file_name')

    await context.bot.send_message(chat_id=update.effective_chat.id, text='Gerando resultado...')
    result = get_alt_text(file.file_path)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=result)

def get_alt_text(image_url): 
    # Run ML Model with imageUrl
    model = replicate.models.get("salesforce/blip")
    version = model.versions.get("2e1dddc8621f72155f24cf2e0adbde548458d3cab9f00c0139eea840d0ac4746")

    # Get the alt text result and return it
    return version.predict(image=image_url)

def main():
    application = ApplicationBuilder().token(BOT_TOKEN).build()
    echo_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), echo)
    image_handler = MessageHandler(filters.PHOTO, image)

    start_handler = CommandHandler('start', start)
    caps_handler = CommandHandler('caps', caps)

    application.add_handler(MessageHandler(filters.Document.ALL, downloader))
    application.add_handler(echo_handler)
    application.add_handler(image_handler)
    application.add_handler(start_handler)
    application.add_handler(caps_handler)

    inline_caps_handler = InlineQueryHandler(inline_caps)
    application.add_handler(inline_caps_handler)

    # Other handlers
    unknown_handler = MessageHandler(filters.COMMAND, unknown)
    application.add_handler(unknown_handler)

    application.run_polling()

if __name__ == '__main__':
    main()