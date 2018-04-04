class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

   @staticmethod
   def purple(text):
      return color.PURPLE+text+color.END

   @staticmethod
   def cyan(text):
      return color.CYAN+text+color.END

   @staticmethod
   def darkcyan(text):
      return color.DARKCYAN+text+color.END

   @staticmethod
   def blue(text):
      return color.BLUE+text+color.END

   @staticmethod
   def green(text):
      return color.GREEN+text+color.END

   @staticmethod
   def yellow(text):
      return color.YELLOW+text+color.END

   @staticmethod
   def red(text):
      return color.RED+text+color.END

   @staticmethod
   def bold(text):
      return color.BOLD+text+color.END

   @staticmethod
   def underline(text):
      return color.UNDERLINE+text+color.END