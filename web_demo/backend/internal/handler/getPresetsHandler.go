// Code scaffolded by goctl. Safe to edit.
// goctl 1.9.2

package handler

import (
	"net/http"

	"backend/internal/logic"
	"backend/internal/svc"
	"github.com/zeromicro/go-zero/rest/httpx"
)

// Get preset prompts
func GetPresetsHandler(svcCtx *svc.ServiceContext) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		l := logic.NewGetPresetsLogic(r.Context(), svcCtx)
		resp, err := l.GetPresets()
		if err != nil {
			httpx.ErrorCtx(r.Context(), w, err)
		} else {
			httpx.OkJsonCtx(r.Context(), w, resp)
		}
	}
}
