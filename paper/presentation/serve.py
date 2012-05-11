import time
import string
import sys
import os
import re
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer

class Handler(BaseHTTPRequestHandler):

    def do_GET(self):

        special = {
                '/': 'index.html',
                }

        path = special.get(self.path, self.path)

        def case_svg():
            mo = re.match(r'(.*?).svg\?[0-9]+', path)
            if mo is None:
                return False
            else:
                f = open(os.curdir + os.sep + mo.group(1) + '.svg')
                self.send_response(200)
                self.send_header('Content-type', 'image/svg+xml')
                self.end_headers()
                self.wfile.write(f.read())
                f.close()
                return True

        try:
            if self.path == '/favicon.ico':
                pass
            elif self.path.endswith('.html'):
                f = open(os.curdir + os.sep + path)
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(f.read())
                f.close()
            elif self.path.endswith('.css'):
                f = open(os.curdir + os.sep + path)
                self.send_response(200)
                self.send_header('Content-type', 'text/css')
                self.end_headers()
                self.wfile.write(f.read())
                f.close()
            elif self.path.endswith('.js'):
                f = open(os.curdir + os.sep + path)
                self.send_response(200)
                self.send_header('Content-type', 'text/javascript')
                self.end_headers()
                self.wfile.write(f.read())
                f.close()
            elif self.path.endswith('thumb.png'):
                pass
            elif self.path.endswith('.png'):
                f = open(os.curdir + os.sep + path)
                self.send_response(200)
                self.send_header('Content-type', 'image/png')
                self.end_headers()
                self.wfile.write(f.read())
                f.close()
            elif case_svg(): pass
            else:
                print 'IGNORING GET', self.path

        except IOError:
            self.send_error(404, 'File not found; %s' % self.path)



def main():
    try:
        server = HTTPServer(('', 8080), Handler)
        print 'Server up'
        server.serve_forever()
    except KeyboardInterrupt:
        print 'ctrl-c, going down'
        server.socket.close()


if __name__ == '__main__':
    sys.exit(main())
