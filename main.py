import ccxt
import pandas as pd
import numpy as np
import os
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import mplfinance as mpf
from zigzag import peak_valley_pivots

# تنظیمات صرافی
exchange = ccxt.kucoin()

# لیست جفت‌ارزها و تایم‌فریم‌ها
symbols = ['BTC/USDT', 'XRP/USDT', 'ETH/USDT', 'DOGE/USDT', 'SOL/USDT', 'HBAR/USDT', 'LTC/USDT', 'SUI/USDT', 'PEPE/USDT', 'ADA/USDT', 'SOLV/USDT', 'BNB/USDT', 'XLM/USDT', 'SHIB/USDT', 'ENA/USDT', 'TRX/USDT', 'LINK/USDT', 'AIXBT/USDT', 'PNUT/USDT', 'WIF/USDT', 'WLD/USDT', 'RUNE/USDT', 'ALGO/USDT', 'AVAX/USDT', 'NEIRO/USDT', 'EOS/USDT', 'VET/USDT', 'NEAR/USDT', 'SAND/USDT', 'MOVE/USDT', 'BONK/USDT', 'FET/USDT', 'APT/USDT', 'DOT/USDT', 'FIL/USDT', 'AAVE/USDT', 'UNI/USDT', 'TAO/USDT', 'FLOKI/USDT', 'USUAL/USDT', 'CRV/USDT', 'PHA/USDT', 'TIA/USDT', 'AMP/USDT', 'PENGU/USDT', 'ARB/USDT', 'INJ/USDT', 'CGPT/USDT', 'SEI/USDT', 'IOTA/USDT', 'BCH/USDT', 'RENDER/USDT', 'ZEN/USDT', 'OP/USDT', 'BIO/USDT', 'TON/USDT', 'JASMY/USDT', 'GMT/USDT', 'ICP/USDT', 'ATOM/USDT', 'PENDLE/USDT', 'ETC/USDT', 'RAY/USDT', 'KDA/USDT', 'RSR/USDT', 'BOME/USDT', 'POL/USDT', 'CFX/USDT', 'GRT/USDT', 'VANA/USDT', 'ZRO/USDT', 'IO/USDT', 'ETHFI/USDT', 'STX/USDT', 'ENS/USDT', 'WBTC/USDT', 'COOKIE/USDT', 'TURBO/USDT', 'DYDX/USDT', 'ZK/USDT', 'SUSHI/USDT', 'JUP/USDT', 'BLZ/USDT', 'LDO/USDT', 'ORDI/USDT', 'NOT/USDT', 'EIGEN/USDT', 'OM/USDT', 'THETA/USDT', 'AR/USDT', 'NEO/USDT', 'COW/USDT', 'APE/USDT', 'MANA/USDT', 'MKR/USDT', 'KAIA/USDT', 'CAKE/USDT', 'ARKM/USDT', 'STRK/USDT', 'MEME/USDT', 'EDU/USDT', 'BAL/USDT', 'XAI/USDT', 'CHZ/USDT', 'MANTA/USDT', 'DASH/USDT', 'FTT/USDT', 'BLUR/USDT', 'PROM/USDT', 'SUPER/USDT', 'EGLD/USDT', 'ROSE/USDT', 'ONE/USDT', 'CETUS/USDT', 'LPT/USDT', 'AGLD/USDT', 'FIDA/USDT', 'XTZ/USDT', 'ALT/USDT', 'LUMIA/USDT', 'W/USDT', 'JTO/USDT', 'AXS/USDT', 'LUNC/USDT', 'PYR/USDT', 'COMP/USDT', 'PEOPLE/USDT', 'ME/USDT', 'DOGS/USDT', 'ZRX/USDT', 'PYTH/USDT', 'LUNA/USDT', 'LQTY/USDT', 'QNT/USDT', 'STG/USDT', 'SNX/USDT', 'BB/USDT', 'IOST/USDT', 'ASTR/USDT', 'TRB/USDT', 'SSV/USDT', 'ZIL/USDT', 'PIXEL/USDT', 'MINA/USDT', 'PORTAL/USDT', 'LISTA/USDT', 'IMX/USDT', 'YFI/USDT', 'CELO/USDT', 'ACX/USDT', 'DEXE/USDT', 'WOO/USDT', '1INCH/USDT', 'CKB/USDT', 'FLOW/USDT', 'SCR/USDT', 'ATA/USDT', 'AEVO/USDT', 'VANRY/USDT', 'ONT/USDT', 'ENJ/USDT', 'ANKR/USDT', 'QKC/USDT', 'MAGIC/USDT', 'COTI/USDT', 'AVA/USDT', 'HMSTR/USDT', 'ZEC/USDT', 'DYM/USDT', 'FXS/USDT', 'METIS/USDT', 'SKL/USDT', 'UTK/USDT', 'GAS/USDT', 'OMNI/USDT', 'QTUM/USDT', 'LRC/USDT', 'KSM/USDT', 'YGG/USDT', 'ACH/USDT', 'GMX/USDT', 'DENT/USDT', 'GLMR/USDT', 'CYBER/USDT', 'POLYX/USDT', 'ACE/USDT', 'TRU/USDT', 'CVX/USDT', 'WIN/USDT', 'JST/USDT', 'HIGH/USDT', 'C98/USDT', 'MBL/USDT', 'REZ/USDT', 'RVN/USDT', 'STORJ/USDT', 'BAT/USDT', 'CATI/USDT', 'CLV/USDT', 'DEGO/USDT', 'XEC/USDT', 'TLM/USDT', 'ID/USDT', 'TNSR/USDT', 'SLP/USDT', 'SXP/USDT', 'ILV/USDT', 'ORCA/USDT', 'FLUX/USDT', 'TFUEL/USDT', 'KAVA/USDT', 'POND/USDT', 'MOVR/USDT', 'MASK/USDT', 'PAXG/USDT', 'BICO/USDT', 'USTC/USDT', 'CHR/USDT', 'ELF/USDT', 'IOTX/USDT', 'API3/USDT', 'NTRN/USDT', 'ICX/USDT', 'HIFI/USDT', 'TWT/USDT', 'DGB/USDT', 'STRAX/USDT', 'OSMO/USDT', 'ALPHA/USDT', 'CVC/USDT', 'NMR/USDT', 'NFP/USDT', 'OGN/USDT', 'SCRT/USDT', 'CELR/USDT', 'RDNT/USDT', 'LSK/USDT', 'UMA/USDT', 'AUCTION/USDT', 'AUDIO/USDT', 'BANANA/USDT', 'ARPA/USDT', 'COMBO/USDT', 'SYN/USDT', 'REQ/USDT', 'SLF/USDT', 'RPL/USDT', 'DUSK/USDT', 'HFT/USDT', 'ALICE/USDT', 'RLC/USDT', 'SUN/USDT', 'GLM/USDT', 'KNC/USDT', 'WAXP/USDT', 'AMB/USDT', 'MAV/USDT', 'G/USDT', 'CTSI/USDT', 'LINA/USDT', 'VIDT/USDT', 'OXT/USDT', 'T/USDT', 'DODO/USDT', 'BAND/USDT', 'MTL/USDT', 'ADX/USDT', 'VOXEL/USDT', 'PUNDIX/USDT', 'TUSD/USDT', 'AERGO/USDT', 'ALPINE/USDT', 'DIA/USDT', 'ERN/USDT', 'BURGER/USDT', 'PERP/USDT', 'NKN/USDT', 'QI/USDT', 'CREAM/USDT', 'SYS/USDT', 'GTC/USDT', 'BSW/USDT', 'XNO/USDT', 'HARD/USDT', 'DCR/USDT', 'LTO/USDT', 'SFP/USDT', 'USDP/USDT', 'FORTH/USDT', 'GNS/USDT', 'DATA/USDT', 'QUICK/USDT', 'KMD/USDT', 'WAN/USDT', 'LOKA/USDT', 'XEM/USDT', 'WAVES/USDT', 'ETHUP/USDT', 'BTCUP/USDT', 'OMG/USDT', 'XMR/USDT', 'HNT/USDT', 'BULL/USDT', 'EPX/USDT', 'POLS/USDT', 'UNFI/USDT', 'BOND/USDT', 'REN/USDT', 'BSV/USDT', 'BTT/USDT', 'REEF/USDT', 'LOOM/USDT', 'KEY/USDT']
timeframes = {"30m": "30m", "1H": "1h", "2H": "2h", "4H": "4h", "1D": "1d"}
num_candles = 100  # تعداد کندل‌ها برای بررسی

# حساسیت ZigZag
zigzag_params = {
    "30m": (0.03, -0.03), "1H": (0.07, -0.07), "2H": (0.1, -0.1),
    "4H": (0.12, -0.12), "1D": (0.2, -0.2)
}

# تنظیمات ایمیل از متغیرهای محیطی
EMAIL_SENDER = os.getenv('EMAIL_SENDER')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')
EMAIL_RECEIVER = os.getenv('EMAIL_RECEIVER')

# نتایج نهایی
results = {}
chart_files = []


def plot_chart(symbol, timeframe):
    """ رسم نمودار قیمت با EMA، MACD و خطوط فیبوناچی و ذخیره‌ی تصویر """
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=120)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        high_price = df['high'].max()
        low_price = df['low'].min()
        log_high_price, log_low_price = np.log(high_price), np.log(low_price)

        levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.764, 1.0]
        fib_levels = {level: np.exp(log_high_price - (log_high_price - log_low_price) * level) for level in levels}

        addplots = [
            mpf.make_addplot(df['MACD'], panel=1, color='black'),
            mpf.make_addplot(df['Signal Line'], panel=1, color='red')
        ]

        for level, value in fib_levels.items():
            addplots.append(mpf.make_addplot([value] * len(df), linestyle='dashed'))

        os.makedirs('charts', exist_ok=True)
        chart_file = f'charts/{symbol.replace("/", "_")}_{timeframe}.png'
        mpf.plot(df, type='candle', addplot=addplots, volume=True, style='yahoo',
                 savefig=dict(fname=chart_file, dpi=100, bbox_inches="tight"),
                 title=f'{symbol} {timeframe}', ylabel='Price (log scale)', yscale='log')
        return chart_file
    except Exception as e:
        print(f"⚠ خطا در رسم نمودار {symbol} {timeframe}: {e}")
        return None


for symbol in symbols:
    for tf_label, tf in timeframes.items():
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=num_candles)

            if not ohlcv or len(ohlcv) < num_candles:
                print(f"⚠ داده‌ای برای {symbol} در تایم‌فریم {tf_label} یافت نشد.")
                continue

            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            up_thresh, down_thresh = zigzag_params[tf_label]
            pivots = peak_valley_pivots(df["close"], up_thresh=up_thresh, down_thresh=down_thresh)
            pivot_indexes = df.index[pivots != 0].tolist()

            if len(pivot_indexes) < 3:
                continue

            last_price = df["close"].iloc[-1]
            highest_idx = df["high"].idxmax()
            last_HH = df["high"].iloc[highest_idx]

            df_before_HH = df.loc[:highest_idx]
            start_Low = df_before_HH["low"].min()

            if start_Low == df_before_HH["low"].min() and highest_idx <= num_candles:
                for i in range(len(pivot_indexes) - 2, 0, -1):
                    first_idx, second_idx, third_idx = pivot_indexes[i - 1], pivot_indexes[i], pivot_indexes[i + 1]
                    first_type, second_type, third_type = pivots[first_idx], pivots[second_idx], pivots[third_idx]

                    if first_type == -1 and second_type == 1:
                        if third_type == -1 and df["high"].iloc[third_idx] < last_HH:
                            last_LH = df["high"].iloc[third_idx]
                            last_LL = df["low"].iloc[third_idx]

                            df_after_LH = df.loc[third_idx:]
                            if (df_after_LH["low"] < start_Low).any():
                                continue

                            if symbol not in results:
                                results[symbol] = []
                            results[symbol].append(tf_label)

                            chart_path = plot_chart(symbol, tf)
                            if chart_path:
                                chart_files.append(chart_path)
                            break

        except Exception as e:
            print(f"⚠ خطا در پردازش {symbol} در تایم‌فریم {tf_label}: {e}")
            continue


def get_total_size(files):
    """محاسبه حجم کل فایل‌ها"""
    total_size = 0
    for file in files:
        total_size += os.path.getsize(file)
    return total_size


def send_email_batch(results, chart_files):
    """ ارسال ایمیل با چندین فایل در صورت نیاز به تقسیم """
    max_size = 20 * 1024 * 1024  # 20MB
    total_size = get_total_size(chart_files)
    batch_number = 1

    # تقسیم فایل‌ها به گروه‌ها
    while total_size > max_size:
        # تقسیم فایل‌ها به گروه‌های کوچکتر
        batch_files = []
        while chart_files and get_total_size(batch_files + [chart_files[0]]) <= max_size:
            batch_files.append(chart_files.pop(0))

        # ارسال ایمیل با فایل‌های گروهی
        send_email(results, batch_files, batch_number)
        batch_number += 1
        total_size = get_total_size(chart_files)  # به‌روزرسانی حجم باقی‌مانده


def send_email(results, chart_files, batch_number=1):
    """ ارسال ایمیل به همراه نتایج و نمودارهای مربوطه """
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER
        msg['Subject'] = f"📈 نتایج تحلیل بازار - بخش {batch_number}"

        body = "نتایج موقعیت‌های پیدا شده:\n\n"
        if results:
            for symbol, timeframes in results.items():
                body += f"{symbol} {' '.join(timeframes)}\n"
        else:
            body += "📌 موقعیتی یافت نشد.\n"

        msg.attach(MIMEText(body, 'plain'))

        for file in chart_files:
            with open(file, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(file)}")
            msg.attach(part)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())

        print(f"✅ ایمیل بخش {batch_number} با موفقیت ارسال شد.")
    except Exception as e:
        print(f"⚠ خطا در ارسال ایمیل: {e}")


if results:
    send_email_batch(results, chart_files)
