import sys
import logging
 
class LoggerWriter:
    def __init__(self, level='log', out_dir='./'):
        # 创建一个logger
        self.logger = logging.getLogger("logger")
        self.logger.setLevel(logging.DEBUG)
        # 创建一个handler，用于写入日志文件
        if not self.logger.handlers: 
            fh = logging.FileHandler(out_dir+'/out.log', encoding='utf-8')
            fh.setLevel(logging.DEBUG)
            # 再创建一个handler，用于输出到控制台
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            # 定义handler的输出格式
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            # 给logger添加handler
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)
 
        # 日志级别
        self.level = level
 
    def write(self, message):
        # 只有message非空时才记录
        if message.rstrip() != "":
            self.logger.log(self.level, message.rstrip())
 
    def flush(self):
        # 这个方法是为了满足文件对象的接口，但是什么都不需要做
        pass
 
    def stdout_to_log_file(self):
        sys.stdout = LoggerWriter(logging.INFO)
        sys.stderr = LoggerWriter(logging.ERROR)
 
