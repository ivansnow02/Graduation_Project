// Code scaffolded by goctl. Safe to edit.
// goctl 1.9.2

package main

import (
	"flag"
	"fmt"
	"net/http"

	"backend/internal/config"
	"backend/internal/handler"
	"backend/internal/svc"

	"github.com/zeromicro/go-zero/core/conf"
	"github.com/zeromicro/go-zero/rest"
)

var configFile = flag.String("f", "etc/arena-api.yaml", "the config file")

func main() {
	flag.Parse()

	var c config.Config
	conf.MustLoad(*configFile, &c)

	server := rest.MustNewServer(c.RestConf,
		rest.WithCustomCors(
			func(header http.Header) {
				header.Set("Access-Control-Allow-Origin", "*")
				header.Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")
				header.Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
			},
			nil,
			"*",
		),
	)
	defer server.Stop()

	ctx := svc.NewServiceContext(c)
	handler.RegisterHandlers(server, ctx)

	fmt.Printf("Starting server at %s:%d...\n", c.Host, c.Port)
	server.Start()
}
