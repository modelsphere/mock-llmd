# mock-llmd: one static Go binary on distroless.

# Multi-arch: Go cross-compiles, so the build stage runs on the builder's own
# platform and only the runtime stage is per-target -- no emulation needed.
FROM --platform=$BUILDPLATFORM golang:1.27 AS build
ARG TARGETOS TARGETARCH

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download

# Only the sources: a locally built ./mock-llmd binary in the checkout stays out.
COPY *.go ./
RUN CGO_ENABLED=0 GOOS="${TARGETOS}" GOARCH="${TARGETARCH}" go build -trimpath -ldflags="-s -w" -o /out/mock-llmd .

FROM gcr.io/distroless/static-debian12:nonroot

COPY --from=build /out/mock-llmd /mock-llmd

USER nonroot:nonroot
# PORT or LISTEN_ADDR override it; the chart's service.port is 8081 too.
EXPOSE 8081

ENTRYPOINT ["/mock-llmd"]
