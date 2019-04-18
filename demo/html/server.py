import bottle
import os
pro_path = os.path.split(os.path.realpath(__file__))[0]
src_path = os.path.join(pro_path, "0824", "src")

@bottle.route('/src/<filename:re:.*\.css|.*\.png|.*\.js|.*\.ico>')
def server_static(filename):
        return bottle.static_file(filename, root=src_path)

#@bottle.route('/src/js/<filename:re:.*\.css|.*\.png|.*\.js|.*\.ico>')
#def server_static(filename):
#        return bottle.static_file(filename, root=src_path)
#
#@bottle.route('/src/css/<filename:re:.*\.css|.*\.png|.*\.js|.*\.ico>')
#def server_static(filename):
#        return bottle.static_file(filename, root=src_path)

@bottle.route('/')
def trans():
    return bottle.static_file('index.html', root="./0824")



bottle.run(host="0.0.0.0", server="cherrypy", port=9091)
