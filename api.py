from chat_delivery_gateway import create_fastapi_app
from handbook_support_bot import build_handbook_bundle


bundle = build_handbook_bundle()
app = create_fastapi_app(bundle.chat_gateway)
